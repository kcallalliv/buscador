# Documentación Técnica Detallada — `app/main_dev.py`

## 1. Objetivo del archivo
`main_dev.py` implementa el endpoint `GET /dev/query` del buscador IA con una lógica extensa de:
- normalización de consultas,
- clasificación de intención (reglas + Gemini),
- búsquedas especializadas en BigQuery,
- orquestación de respuestas comerciales,
- integración con Vertex para consultas generales,
- fallback inteligente a películas,
- trazabilidad y persistencia en BigQuery.

Es el entorno de experimentación/iteración avanzada antes de llevar cambios a producción.

---

## 2. Configuración y bootstrap local del módulo

### 2.1 Inicialización IA
- `vertexai_init(project, location)`.
- modelo Gemini configurable por env:
  - `VERTEX_PROJECT`
  - `VERTEX_LOCATION`
  - `GEMINI_MODEL_NAME`
- `GenerationConfig` orientado a salida JSON (`response_mime_type="application/json"`).

### 2.2 Integraciones externas
- `CLAROVIDEO_MOVIES_API_URL` para búsqueda de películas por API.
- cliente BigQuery global del módulo.

### 2.3 Blueprint y CORS de desarrollo
- `dev_bp = Blueprint("dev_bp", __name__)`
- CORS restringido a orígenes de pruebas y dominios autorizados.

---

## 3. Responsabilidades funcionales del archivo

1. **Preclasificar** consultas (celulares, planes, tienda_productos, pelicula, general).
2. **Resolver intención** con la mejor fuente:
   - celulares: búsquedas/ranking BQ + copy comercial,
   - películas: catálogo Claro Video,
   - general: Vertex Discovery Engine,
   - planes/accesorios: queries especializadas.
3. **Mantener experiencia comercial**:
   - títulos/descripciones dinámicas,
   - recomendados fijos o inferidos.
4. **Persistir auditoría**:
   - historial de consultas,
   - cache de respuestas definitivas para `general`.

---

## 4. Secciones lógicas del archivo

## 4.1 Detección rápida de títulos de películas
Funciones:
- `looks_like_title`
- `try_movies_first`

Aplican filtros para decidir si la consulta parece título de entretenimiento y, de ser así, llaman a Movies API para respuesta temprana.

## 4.2 Normalización textual
Funciones clave:
- `sanitize_query`
- `strip_accents`
- `roman_to_arabic`
- `_fix_special_typos`
- `normalize_query_local`

Objetivo: estandarizar consulta ante errores ortográficos, acentos, unicode raro, números romanos y variaciones frecuentes.

## 4.3 Clasificación de intención
Función principal:
- `classify_with_gemini`

Estrategia híbrida:
1. Fast-path por regex/modelos móviles y términos fuertes.
2. Si no hay certeza, invoca Gemini con prompt categórico.
3. Aplica overrides locales para evitar falsos positivos.

Categorías soportadas:
- `celulares`
- `tienda_productos`
- `planes`
- `pelicula`
- `general`

## 4.4 Utilitarios de parsing para celulares
Bloque especializado con funciones como:
- `_color_from_query`, `_extract_color_from_title`
- `_extract_storage_from_title`, `_storage_from_query`
- `_extract_model_number_from_title`
- `_variant_rank`
- `_display_query_for_phone`

Estas funciones incrementan la precisión de matching y presentación comercial cuando no existe coincidencia exacta.

## 4.5 Motor de búsqueda de productos en BigQuery
Funciones:
- `search_products_bq`
- `search_brand_only_bq`
- `search_brand_top5_products`
- `_run_accessories_query`
- `search_accessories_bq`
- `search_plans_bq`

Incluye SQL de scoring complejo para:
- familia y variante (Pro/Plus/Ultra/SE/FE/Lite),
- capacidad (GB),
- modalidad comercial (renovación/portabilidad/línea nueva/liberado),
- jerarquías de ranking por reglas de negocio.

## 4.6 Construcción de respuesta de películas
Función:
- `build_claro_video_response`

Implementa:
- consulta a catálogo BQ de Claro Video,
- armado de `listado` y `relacionados`,
- truncado de descripciones,
- fallback de contenido cuando no hay resultados.

## 4.7 Endpoint HTTP
Función:
- `query_dev` (ruta `/dev/query`, método `GET`, limit `10/min`).

Orquesta todo el pipeline end-to-end.

---

## 5. Flujo completo de `query_dev`

1. **Lectura de entrada**: `q`, `sia_id`, `User-Agent`, timestamp.
2. **Validación básica**: si `q` vacío -> `400`.
3. **Normalización y clasificación**: `classify_with_gemini`.
4. **Consulta de cache** (`buscar_respuesta_definitiva`) cuando aplica.
5. **Ruteo por categoría**:
   - `pelicula` -> `build_claro_video_response`.
   - `celulares` -> búsqueda/ranking en BQ y armado comercial.
   - `tienda_productos` -> accesorios/productos + recomendados.
   - `planes` -> resultados de planes.
   - `general` -> `get_summary_from_vertex` + guarda en cache.
6. **Enriquecimiento**:
   - agrega `_meta` (`normalized_query`, `category`, `cache_hit`).
   - guarda historial.
7. **Fallback final a películas** cuando respuesta general no es satisfactoria.
8. **Retorno HTTP** con JSON final.
9. **Error global**: captura excepción y responde `500` con detalle controlado.

---

## 6. Reglas comerciales destacadas

1. Priorización de respuestas de venta para celulares.
2. Uso de templates para títulos/descripciones según:
   - coincidencia exacta,
   - modelo similar,
   - no encontrado.
3. Recomendados contextuales (accesorios por marca/consulta) o fallback fijo.
4. Separación entre intención de entretenimiento y telco para reducir confusión de dominio.

---

## 7. Persistencia y trazabilidad

`main_dev.py` usa utilitarios de `app/common.py` para:
- **historial**: `guardar_pregunta_en_historial`.
- **cache general**: `guardar_en_respuestas_definitivas` y `buscar_respuesta_definitiva`.

Esto permite auditoría de uso y mejora de performance para preguntas repetidas.

---

## 8. Integraciones externas del flujo DEV

1. **BigQuery**: principal fuente estructurada de productos, top ventas, catálogo video.
2. **Gemini (Vertex AI)**: clasificación de intención cuando fast-path no basta.
3. **Discovery Engine**: respuestas generales de contenido/no transaccionales.
4. **ClaroVideo Movies API**: fallback/atajo para títulos de entretenimiento.

---

## 9. Diferencias técnicas clave frente a PROD

1. `main_dev.py` contiene más funciones de parsing fino y copy dinámico.
2. El ruteo celulares se apoya más en lógica local y experimentación.
3. Mayor densidad de reglas de negocio y heurísticas iterativas.
4. CORS admite dominios de test adicionales.

---

## 10. Riesgos técnicos en `main_dev.py`

1. **Tamaño y complejidad altos** (>2.5k líneas).
2. **Duplicación parcial con `main_prod.py`**.
3. **SQL embebidos largos** difíciles de testear/versionar.
4. **Lógica de negocio dispersa** entre constantes, regex y branches.

---

## 11. Recomendaciones de refactor

1. Extraer capas por dominio:
   - normalización,
   - clasificación,
   - repositorios BQ,
   - render de respuesta.
2. Unificar contrato de salida con validación de esquema.
3. Mover SQL a archivos versionados y testeables.
4. Crear pruebas unitarias para funciones puras de parsing/ranking.
5. Crear pruebas de integración por categoría de intención.

---

## 12. Inventario funcional detallado (qué hace cada función)

### 12.1 Detección inicial y llamadas externas
- `looks_like_title`: determina si la consulta parece un título de entretenimiento (longitud, ruido telco, marcas bloqueadas).
- `try_movies_first`: intenta resolver rápidamente la consulta contra Movies API cuando la query parece título.
- `call_clarovideo_movies_api`: encapsula la llamada HTTP POST al endpoint de películas y retorna JSON.
- `title_is_clarovideo_disfruta`: detecta patrón de título genérico de respuesta orientada a Claro Video.
- `description_is_clarovideo_disfruta`: detecta frases genéricas en título usadas para fallback a películas.
- `description_is_clarovideo_genero`: detecta cuando el título sugiere clasificación por género de películas.
- `call_gemini_json`: ejecuta Gemini con prompt + input, espera JSON y aplica fallback seguro si no parsea.

### 12.2 Normalización y saneamiento de consulta
- `roman_to_arabic`: convierte numerales romanos frecuentes (xiv–xx) a arábigos para mejorar matching de modelos.
- `strip_accents`: elimina tildes/diacríticos para comparar textos de forma uniforme.
- `sanitize_query`: limpia unicode/control chars, normaliza espacios y devuelve string seguro.
- `_fix_special_typos`: corrige errores ortográficos puntuales difíciles con reglas simples.
- `normalize_query_local`: pipeline completo de normalización (sanitize + lowercase + typos + romanización).
- `_safe_pick_normalized`: decide si confiar en la normalización del modelo o mantener la local según similitud.
- `_match_any`: helper genérico para evaluar múltiples regex contra un texto.

### 12.3 Búsqueda de catálogo de Claro Video
- `search_claro_video_catalog`: consulta BigQuery de catálogo, calcula score por similitud y arma principales/relacionados.
- `build_claro_video_response`: convierte resultados de catálogo en payload final de negocio (`Found/NotFound`).

### 12.4 Clasificación de intención
- `classify_with_gemini`: clasifica la consulta en `celulares/tienda_productos/planes/pelicula/general` con fast-path y overrides.

### 12.5 Utilitarios léxicos y extracción de señales
- `_terms_from_query`: tokeniza términos alfanuméricos útiles para filtros.
- `_nums_from_query`: extrae números relevantes de la consulta (modelos/capacidad/precio).
- `_brand_terms_from_query`: identifica términos de marca explícitos en la consulta.
- `_canonical_brand_from_query`: mapea variantes de marca a una marca canónica.
- `_strip_fillers_from_query`: elimina palabras de relleno que no aportan al matching.
- `_color_from_query`: detecta color pedido y retorna query limpia + color.
- `_extract_color_from_title`: extrae color detectado desde título de producto.
- `_extract_storage_from_title`: obtiene capacidad (GB) desde título.
- `_extract_model_number_from_title`: extrae número de modelo según marca/familia.
- `_variant_rank`: puntúa variante comercial/técnica (pro, plus, ultra, etc.).
- `_brand_display_label`: construye etiqueta comercial de marca para textos de salida.
- `_format_phone_label`: normaliza formato visual de nombre de equipo.
- `_display_query_for_phone`: genera versión “amigable” de query para títulos/descripciones.

### 12.6 Generación de copy comercial
- `_pick_phone_title`: selecciona template de título para respuesta de equipos según contexto.
- `_pick_phone_desc`: selecciona template de descripción para respuesta de equipos.
- `_pick_brand_description`: genera descripción orientada a marca cuando la consulta es brand-only.

### 12.7 Reglas de negocio de celulares
- `_is_brand_only_query`: determina si la consulta es solo marca (sin modelo/atributos).
- `_has_prepago_term`: detecta intención explícita de prepago.
- `_explicit_modality_from_query`: detecta modalidad pedida (portabilidad, renovación, etc.).
- `_storage_from_query`: detecta capacidad solicitada por el usuario.
- `_query_mentions_economy`: detecta intención de bajo precio/economía.
- `_extract_main_number`: obtiene número principal de la consulta para ranking.
- `_samsung_series_from_query`: identifica familia Samsung (S/A/M/Z) para afinamiento.
- `_pick_by_modality_priority`: reordena resultados según prioridad comercial de modalidad.
- `_category_hint_from_query`: sugiere subcategoría de accesorio/producto para acotar búsqueda.

### 12.8 Acceso a datos y ranking en BigQuery
- `search_products_bq`: consulta principal de equipos celulares con scoring complejo y filtros por intención.
- `search_brand_only_bq`: obtiene catálogo cuando la consulta es de marca sin modelo específico.
- `_last_closed_month_start_date_lima`: calcula inicio del último mes cerrado en zona horaria local.
- `_month_spanish_name`: convierte número de mes a nombre en español para copy.
- `_fetch_top5_names_by_brand_last_month`: trae top 5 vendidos de marca en último mes cerrado.
- `search_brand_top5_products`: resuelve productos top por marca para respuesta comercial destacada.
- `_run_accessories_query`: ejecuta SQL de accesorios con combinaciones de filtros de marca/categoría.
- `search_accessories_bq`: orquesta estrategia de búsqueda de accesorios con relajación progresiva.
- `search_plans_bq`: consulta y ordena planes móviles/hogar según señales de consulta.

### 12.9 Endpoint y orquestación final
- `query_dev`: controlador HTTP principal; valida entrada, clasifica, enruta, persiste historial, arma respuesta y maneja errores.

---

## 13. Resumen ejecutivo

`main_dev.py` funciona como laboratorio avanzado del buscador IA: combina heurística, IA generativa y SQL de negocio para resolver consultas heterogéneas con foco en conversión comercial y robustez operacional.