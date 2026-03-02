# Anexo 06 — Fuentes de datos y trazabilidad

## 1. Objetivo
Documentar de forma explícita **de dónde sale la información** que devuelve la API, qué componente la consume, su criticidad y el comportamiento de fallback.

---

## 2. Matriz maestra de fuentes

| Fuente | Tipo | Datos que aporta | Consumido por | Criticidad | Fallback si falla |
|---|---|---|---|---|---|
| `prd-claro-mktg-data-storage.tienda_claro.productos` | BigQuery | Catálogo de productos/equipos, atributos comerciales, modalidad | `search_products_bq`, `search_brand_only_bq`, `search_accessories_bq`, `search_plans_bq` | Alta | Respuesta "Not Found"/recomendados fijos o paso a flujo general según categoría |
| `prd-claro-mktg-data-storage.master_analytics.sales_top5_brand_monthly` | BigQuery | Top 5 más vendidos por marca y mes | `_fetch_top5_names_by_brand_last_month`, `search_brand_top5_products` | Media-Alta | Se omite bloque top5 y se continúa con búsqueda general de productos |
| `prd-claro-mktg-data-storage.claro_video.catalogo` | BigQuery | Catálogo de películas/series, metadata y URLs | `search_claro_video_catalog`, `build_claro_video_response` | Alta | Respuesta de películas en modo `NotFound` con enlaces genéricos de Claro Video |
| `prd-claro-mktg-data-storage.claro_searchai_logs.historial_preguntas` | BigQuery | Auditoría de consultas y respuestas emitidas | `guardar_pregunta_en_historial` | Media | La API puede responder aunque falle logging (impacto en trazabilidad) |
| `prd-claro-mktg-data-storage.claro_searchai_logs.respuestas_definitivas` | BigQuery | Cache de respuestas generales reutilizables | `buscar_respuesta_definitiva`, `guardar_en_respuestas_definitivas` | Media | Se recalcula respuesta por Vertex (sin cache hit) |
| `TIENDA_API_URL` (Cloud Run) | API HTTP externa | Resultado de búsqueda de celulares/equipos (flujo productivo) | `call_tienda_api` (prod) | Alta | Respuesta de error controlada + recomendados fijos |
| `CLAROVIDEO_MOVIES_API_URL` | API HTTP externa | Búsqueda rápida de títulos de entretenimiento | `try_movies_first`, `call_clarovideo_movies_api` | Media-Alta | Continúa flujo de catálogo BQ/Vertex según categoría |
| Discovery Engine (`PROJECT_ID`, `LOCATION`, `ENGINE_ID`) | Servicio AI externo | Respuesta general estructurada (`titulo`, `descripcion`, `listado`, `relacionados`) | `get_summary_from_vertex` | Alta | JSON amigable por defecto y/o fallback de flujo según endpoint |
| Gemini (`GEMINI_MODEL_NAME`) | Servicio AI externo | Clasificación de intención y normalización sugerida | `classify_with_gemini`, `call_gemini_json` | Media | Fast-path local + categoría `general` en caso de falla |

---

## 3. Trazabilidad por ruta de negocio

## 3.1 `/prod/query`
1. Entrada y clasificación (`classify_with_gemini`) con soporte Gemini + reglas locales.
2. Si `celulares`, prioriza `TIENDA_API_URL` y estrategias de marca/top5 en BigQuery.
3. Si `pelicula`, usa catálogo `claro_video.catalogo` y fallback de Movies API.
4. Si `general`, usa Discovery Engine y cache en `respuestas_definitivas`.
5. Siempre registra historial en `historial_preguntas`.

## 3.2 `/dev/query`
1. Clasificación híbrida (reglas + Gemini).
2. Ruteo más extenso con consultas BigQuery para celulares/accesorios/planes.
3. En `general`, invoca Discovery Engine y cachea respuestas.
4. Aplica fallback a películas cuando respuesta general no satisface criterios.

---

## 4. Fuente de verdad por dominio

- **Celulares (PROD)**: `TIENDA_API_URL` como fuente principal en búsquedas de modelos.
- **Celulares (DEV)**: BigQuery (`tienda_claro.productos`) + lógica de ranking local.
- **Top vendidos por marca**: `sales_top5_brand_monthly`.
- **Películas/series**: `claro_video.catalogo` (+ Movies API para detección rápida).
- **Consultas generales**: Discovery Engine (con prompt y parseo estructurado).
- **Persistencia/caché**: tablas `historial_preguntas` y `respuestas_definitivas`.

---

## 5. Campos sugeridos de auditoría de fuente

Para mejorar trazabilidad operativa en respuestas y logs, agregar:
- `data_source_primary` (ej. `tienda_api`, `bq_productos`, `discovery_engine`)
- `data_source_secondary` (ej. `movies_api`, `bq_top5`)
- `source_latency_ms`
- `source_status` (`ok`, `timeout`, `error`, `fallback`)
- `fallback_applied` (`true/false`)

---

## 6. Riesgos de origen de datos

1. **Divergencia entre fuentes** (ej. Tienda API vs BQ productos en DEV).
2. **Cambios de esquema BQ** sin coordinación.
3. **Timeouts de servicios externos** (Vertex/Tienda/Movies API).
4. **Dependencia de modelo** para formato JSON en respuestas generales.

Mitigación recomendada:
- Monitoreo por fuente.
- Versionado de contrato de salida.
- Pruebas de regresión por categoría e integración.

---

## 7. Checklist de validación de fuentes (pre-release)

1. Validar acceso y permisos a cada tabla BQ.
2. Probar reachability de `TIENDA_API_URL` y `CLAROVIDEO_MOVIES_API_URL`.
3. Verificar `PROJECT_ID/ENGINE_ID` en Discovery Engine.
4. Ejecutar smoke tests por categoría con trazabilidad de fuente esperada.
5. Confirmar escritura en `historial_preguntas` y lectura/escritura de `respuestas_definitivas`