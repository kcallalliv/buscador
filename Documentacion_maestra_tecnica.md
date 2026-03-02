# Documento Maestro Técnico — Buscador IA Claro

## 1) Objetivo del sistema
Este servicio expone un buscador inteligente vía API Flask para responder consultas de usuarios de Claro Perú, enrutar búsquedas a catálogos (celulares/accesorios/planes), consultar servicios externos (Tienda API y Claro Video), apoyarse en Vertex AI/Discovery Engine para respuestas generales y persistir historial/cache en BigQuery.

---

## 2) Inventario del repositorio

- `app/__init__.py`: bootstrap de Flask, CORS global, rate limit base y registro de blueprints `/prod` y `/dev`.
- `app/main_prod.py`: lógica productiva completa del endpoint `GET /prod/query`.
- `app/main_dev.py`: versión extendida para `GET /dev/query` (más heurísticas y copy comercial).
- `app/common.py`: utilitarios transversales (control de origen, cache y auditoría en BigQuery).
- `app/vertex_handler.py`: integración con Discovery Engine (AnswerQuery) + fallback JSON estructurado.
- `app/extensions.py`: instancia alternativa de `Limiter` (actualmente no es la usada por el bootstrap principal).
- `requirements.txt`: dependencias Python.
- `Dockerfile`: empaquetado para despliegue en Cloud Run con Gunicorn.

---

## 3) Arquitectura funcional (alto nivel)

1. **Entrada HTTP** (`/prod/query` o `/dev/query`) recibe `q` (consulta), `sia_id` y `User-Agent`.
2. **Normalización + clasificación de intención**:
   - Reglas locales (regex, diccionarios, fast-path).
   - Gemini (clasificación JSON) como refuerzo.
3. **Enrutamiento por intención**:
   - `celulares`: Tienda API (prod) o BigQuery/heurísticas avanzadas (dev).
   - `tienda_productos`: búsqueda de accesorios/productos por BigQuery.
   - `planes`: consulta de planes por BigQuery.
   - `pelicula`: búsqueda optimizada en catálogo Claro Video por BigQuery y/o API de películas.
   - `general`: Vertex Discovery Engine.
4. **Post-proceso**:
   - Fallback a películas cuando respuesta general no es satisfactoria.
   - Armado de `_meta` (query normalizada, categoría, cache hit).
5. **Persistencia**:
   - Historial de pregunta/respuesta.
   - Cache de respuestas “definitivas” para preguntas generales.

---

## 4) Componentes técnicos

### 4.1 API y runtime
- Framework: Flask.
- Servidor WSGI: Gunicorn (`-w 2 --timeout 55`).
- CORS: habilitado globalmente y reforzado por validación explícita de `Origin`.
- Rate limiting:
  - global: `100 per hour` (por `sia_id` o IP).
  - endpoint `/query`: `10 per minute`.

### 4.2 IA/ML y motores de búsqueda
- **Vertex AI Gemini**: clasificación de intención con salida JSON estricta.
- **Vertex Discovery Engine**: generación de respuestas generales con prompt y política de reintentos.
- **RapidFuzz / SequenceMatcher**: scoring y validación de similitud textual.

### 4.3 Datos
- **BigQuery** se usa para:
  - Catálogo de productos (`tienda_claro.productos`).
  - Top5 de ventas por marca/mes (`master_analytics.sales_top5_brand_monthly`).
  - Catálogo Claro Video (`claro_video.catalogo`).
  - Logs y cache (`claro_searchai_logs.historial_preguntas` y `respuestas_definitivas`).

### 4.4 Servicios externos
- **Tienda API** (Cloud Run): búsqueda de equipos para flujo productivo de celulares.
- **ClaroVideo Movies API**: búsqueda dedicada de películas cuando la consulta parece título.

---

## 5) Contratos de entrada/salida

## 5.1 Endpoint principal
- Método: `GET`
- Rutas:
  - `/prod/query`
  - `/dev/query`
- Parámetros:
  - `q` (obligatorio): texto de consulta.
  - `sia_id` (opcional): identificador de sesión/usuario para rate limit y trazabilidad.

## 5.2 Estructuras de respuesta
Se retornan JSON con estructura variable por tipo, pero normalmente con:
- `titulo`
- `descripcion`
- `status` (`Found`, `NotFound`, `Not Found`, `Error`)
- `producto` y/o `listado`
- `recomendados` y/o `relacionados`
- `tipo_respuesta`
- `query`
- `_meta` (telemetría interna)

---

## 6) Flujo detallado de `/prod/query`

1. Valida origen (`origin_check`) con allowlist.
2. Valida `q` no vacío.
3. Normaliza y clasifica consulta (`classify_with_gemini`).
4. Busca en cache de respuestas definitivas si aplica (flujo general).
5. Enrutamiento principal:
   - **Películas**: construcción dedicada con `build_claro_video_response`.
   - **Celulares**:
     - Si es consulta de marca pura -> Top5 por marca con BQ.
     - Caso general -> `call_tienda_api`.
   - **General**: `get_summary_from_vertex` y posible persistencia en cache.
6. Guarda historial y agrega `_meta`.
7. Ejecuta fallback a películas si respuesta general no fue útil.
8. Devuelve JSON final; en excepción, responde `500` con detalle.

---

## 7) Flujo detallado de `/dev/query`

Misma base conceptual de prod, con diferencias clave:
- Mayor número de utilitarios de parsing (color, capacidad, variante, modelo).
- Funciones para copy comercial dinámico (templates de título/descripción).
- Estrategia de matching más extensa para selección de equipos similares.
- Permite experimentar ajustes de ranking antes de promoverlos a producción.

---

## 8) Clasificación de intención

El sistema usa combinación de:
- **Heurísticas determinísticas**:
  - regex para modelos de teléfono,
  - sets de términos por dominio (planes, accesorios, signals de tienda, hints de películas),
  - normalización agresiva (acentos, roman numerals, typos frecuentes).
- **Gemini**:
  - prompt categórico con salida JSON: `celulares | tienda_productos | planes | pelicula | general`.
- **Overrides finales**:
  - fuerza categoría cuando detecta evidencia fuerte local (modelo/categoría exacta).

Esto reduce latencia/costo (fast-path) y aumenta robustez frente a ruido ortográfico.

---

## 9) Búsquedas y ranking

### 9.1 Claro Video
- Query BigQuery con filtro regex por tokens de consulta.
- Scoring híbrido sobre título/título original/slug.
- Selección de principales y relacionados + backfill por sección si faltan relacionados.

### 9.2 Celulares
- En prod se delega principalmente al microservicio Tienda API para modelos.
- Para marca pura se consulta TOP5 mensual en BigQuery.
- Se ordena por modalidad prioritaria (renovación, portabilidad, línea nueva, liberado) y reglas comerciales.

### 9.3 Accesorios / tienda_productos
- Query por términos con filtros de categoría/brand/strict brand.
- Estrategia incremental de relajación si no hay resultados.

### 9.4 Planes
- Query dedicada sobre campos de plan/línea/precio.
- Enriquecimiento de respuesta con recomendados fijos.

---

## 10) Integración con Vertex Discovery Engine

`get_summary_from_vertex`:
- Configura cliente con endpoint regional/global.
- Construye prompt estricto de JSON.
- Define política de retry para errores transitorios (503/500/timeout).
- Limpia posible envoltura markdown en la respuesta del modelo.
- Parsea JSON del modelo o usa fallback amigable predefinido.
- Extrae citas y las agrega a `listado` con HTML enlazado.

---

## 11) Persistencia y observabilidad

## 11.1 Historial de preguntas
Tabla: `claro_searchai_logs.historial_preguntas`.
- Campos registrados: pregunta normalizada, `sia_id`, timestamps, `user_agent`, SO detectado, JSON de respuesta.

## 11.2 Respuestas definitivas (cache)
Tabla: `claro_searchai_logs.respuestas_definitivas`.
- Permite reutilizar respuestas generales previas para consultas equivalentes.

## 11.3 Metadata operativa
Campo `_meta` en respuesta final:
- `normalized_query`
- `category`
- `cache_hit`

---

## 12) Seguridad y control

- CORS por blueprint y validación explícita de origen.
- Rate limit por `sia_id` o IP.
- Timeouts en llamadas HTTP externas.
- Manejo de excepciones con fallback amigable en varios puntos críticos.

---

## 13) Configuración por variables de entorno

- `VERTEX_PROJECT`
- `VERTEX_LOCATION`
- `GEMINI_MODEL_NAME`
- `CLAROVIDEO_MOVIES_API_URL`
- `TIENDA_API_URL`
- `PORT` (en contenedor)

Valores por defecto están hardcodeados para entorno Cloud Run/Claro.

---

## 14) Despliegue

- Imagen base: `python:3.10-slim`.
- Instala dependencias desde `requirements.txt`.
- Ejecuta `gunicorn` sobre `app:app`.
- Diseñado para Cloud Run (`PORT=8080`).

---

## 15) Diferencias principales DEV vs PROD

1. **Complejidad de matching**: DEV incluye más funciones de parsing/ranking textual.
2. **Copy de respuesta**: DEV tiene más templates comerciales dinámicos.
3. **Canal celulares**:
   - PROD favorece microservicio Tienda API como fuente de verdad.
   - DEV ensaya más lógica local para construir respuesta de equipos.
4. **CORS**: DEV permite más orígenes de prueba.

---

## 16) Riesgos técnicos detectados (para traspaso)

1. **Duplicación alta de lógica** entre `main_dev.py` y `main_prod.py` (coste de mantenimiento).
2. **Acoplamiento a BigQuery schemas** mediante SQL largos embebidos.
3. **Dependencia de prompts largos** para estructura JSON (riesgo de deriva del modelo).
4. **Uso de texto HTML dentro de payloads** en citas (`listado`), requiere cuidado en frontend.
5. **`extensions.py` redundante** frente a `Limiter` definido en `app/__init__.py`.

---

## 17) Recomendaciones de continuidad

1. Extraer lógica compartida DEV/PROD a módulos de dominio (clasificación, búsqueda, render de respuesta).
2. Versionar SQL fuera de código (plantillas o capa repository).
3. Unificar contrato de respuesta (schema JSON formal + validación).
4. Instrumentar métricas (latencia por ruta, cache hit ratio, tasa fallback a películas, errores por dependencia).
5. Añadir suite de pruebas automáticas sobre:
   - normalización,
   - clasificación,
   - ranking,
   - estructura JSON de salida.

---

## 18) Índice técnico de funciones (inventario para handover)

### 18.1 `app/main_prod.py`
`looks_like_title`, `try_movies_first`, `call_clarovideo_movies_api`, `call_tienda_api`, `title_is_clarovideo_disfruta`, `description_is_clarovideo_disfruta`, `description_is_clarovideo_genero`, `call_gemini_json`, `roman_to_arabic`, `strip_accents`, `sanitize_query`, `_fix_special_typos`, `normalize_query_local`, `_safe_pick_normalized`, `_match_any`, `search_claro_video_catalog`, `classify_with_gemini`, `_terms_from_query`, `_nums_from_query`, `_brand_terms_from_query`, `_canonical_brand_from_query`, `_is_brand_only_query`, `_has_prepago_term`, `_explicit_modality_from_query`, `_storage_from_query`, `_query_mentions_economy`, `_extract_main_number`, `_samsung_series_from_query`, `_pick_by_modality_priority`, `_category_hint_from_query`, `search_products_bq`, `_last_closed_month_start_date_lima`, `_month_spanish_name`, `_fetch_top5_names_by_brand_last_month`, `search_brand_top5_products`, `_run_accessories_query`, `search_accessories_bq`, `search_plans_bq`, `build_claro_video_response`, `query_prod`.

### 18.2 `app/main_dev.py`
`looks_like_title`, `try_movies_first`, `call_clarovideo_movies_api`, `title_is_clarovideo_disfruta`, `description_is_clarovideo_disfruta`, `description_is_clarovideo_genero`, `call_gemini_json`, `roman_to_arabic`, `strip_accents`, `sanitize_query`, `_fix_special_typos`, `normalize_query_local`, `_safe_pick_normalized`, `_match_any`, `search_claro_video_catalog`, `classify_with_gemini`, `_terms_from_query`, `_nums_from_query`, `_brand_terms_from_query`, `_canonical_brand_from_query`, `_strip_fillers_from_query`, `_color_from_query`, `_extract_color_from_title`, `_extract_storage_from_title`, `_extract_model_number_from_title`, `_variant_rank`, `_brand_display_label`, `_format_phone_label`, `_display_query_for_phone`, `_pick_phone_title`, `_pick_phone_desc`, `_pick_brand_description`, `_is_brand_only_query`, `_has_prepago_term`, `_explicit_modality_from_query`, `_storage_from_query`, `_query_mentions_economy`, `_extract_main_number`, `_samsung_series_from_query`, `_pick_by_modality_priority`, `_category_hint_from_query`, `search_products_bq`, `search_brand_only_bq`, `_last_closed_month_start_date_lima`, `_month_spanish_name`, `_fetch_top5_names_by_brand_last_month`, `search_brand_top5_products`, `_run_accessories_query`, `search_accessories_bq`, `search_plans_bq`, `build_claro_video_response`, `query_dev`.

### 18.3 `app/common.py`
`extraer_sistema_operativo`, `origin_check`, `buscar_respuesta_definitiva`, `guardar_en_respuestas_definitivas`, `guardar_pregunta_en_historial`.

### 18.4 `app/vertex_handler.py`
`get_summary_from_vertex`.

---

## 19) Comandos de operación rápida

- Desarrollo local (referencial):
  - `pip install -r requirements.txt`
  - `gunicorn -b :8080 -w 2 --timeout 55 "app:app"`
- Endpoints:
  - `GET /prod/query?q=<texto>&sia_id=<id>`
  - `GET /dev/query?q=<texto>&sia_id=<id>`

---

## 20) Estado de documentación base

- `README.md` se encuentra vacío; este documento puede actuar como base inicial del traspaso técnico.

---

## 21) Anexos operativos y de continuidad

Para completar el handover operativo se agregan los siguientes anexos:

1. `ANEXO_01_RUNBOOK_OPERATIVO.md`
2. `ANEXO_02_CONTRATO_API_Y_EJEMPLOS.md`
3. `ANEXO_03_OBSERVABILIDAD_Y_ALERTAS.md`
4. `ANEXO_04_MATRIZ_TROUBLESHOOTING.md`
5. `ANEXO_05_OPERACION_POR_ENTORNO_Y_RELEASE.md`