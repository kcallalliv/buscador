# Anexo 09 — Diccionario de datos de las fuentes

## 1. Objetivo
Definir los campos clave (diccionario de datos) de cada fuente utilizada por el buscador para facilitar mantenimiento, validaciones y traspaso.

> Alcance: campos relevantes para la API (no necesariamente 100% de columnas físicas de cada tabla/origen).

---

## 2. BigQuery — `tienda_claro.productos`
**Uso principal:** catálogo de celulares, accesorios y planes en búsquedas comerciales.

| Campo | Tipo (referencial) | Descripción funcional | Uso en API |
|---|---|---|---|
| `id` | STRING | Identificador único del producto/registro | Deduplicación y trazabilidad en listados |
| `title_product` | STRING | Nombre comercial del equipo/producto | Matching, ranking, título mostrado |
| `title_plan` | STRING | Nombre del plan asociado | Ranking por modalidad/precio |
| `price` / `price_offer` | NUMERIC/STRING | Precio regular/oferta | Orden y copy comercial |
| `brand` | STRING | Marca normalizada/comercial | Filtros por marca |
| `line` | STRING | Línea/modalidad (postpago, prepago, negocios, etc.) | Priorización de resultados |
| `url` | STRING | URL de destino del producto | Navegación del usuario |
| `image` / `image_url` | STRING | Imagen principal | Render en frontend |
| `capacity` (derivable) | STRING/INT | Capacidad de almacenamiento (GB) | Matching por capacidad |
| `color` (derivable) | STRING | Color del equipo | Afinamiento de coincidencia |
| `stock` (si aplica) | BOOL/INT | Disponibilidad | Control de recomendación |

**Observación:** `main_dev.py` deriva muchos atributos desde `title_product` con regex (capacidad, variante, color, modelo).

---

## 3. BigQuery — `master_analytics.sales_top5_brand_monthly`
**Uso principal:** top de equipos más vendidos por marca y mes cerrado.

| Campo | Tipo (referencial) | Descripción funcional | Uso en API |
|---|---|---|---|
| `month` / `period` | DATE/STRING | Mes de agregación | Selección de último mes cerrado |
| `brand` | STRING | Marca agregada | Filtro por marca consultada |
| `product_name` | STRING | Nombre del equipo top | Construcción de lista destacada |
| `sales` / `units` | INT64 | Volumen vendido | Orden de top |
| `rank` | INT64 | Posición del equipo en el top | Corte top N |

---

## 4. BigQuery — `claro_video.catalogo`
**Uso principal:** resultados de entretenimiento (películas/series).

| Campo | Tipo (referencial) | Descripción funcional | Uso en API |
|---|---|---|---|
| `id` | STRING | Identificador de contenido | Deduplicación |
| `section` | STRING | Sección/categoría del catálogo | Backfill de relacionados |
| `title` | STRING | Título principal | Matching y nombre mostrado |
| `title_original` | STRING | Título original | Matching secundario |
| `year` | INT64 | Año de estreno | Contexto del contenido |
| `duration` | STRING/INT | Duración | Contexto del contenido |
| `description` | STRING | Descripción corta | Texto de listado |
| `description_large` | STRING | Descripción larga | Texto enriquecido (con truncado) |
| `rating_code` | STRING | Clasificación de contenido | Contexto |
| `title_uri` | STRING | URI interna del título | Relación catálogo |
| `url` | STRING | Enlace público | Navegación del usuario |
| `image_small` | STRING | Imagen pequeña | Render fallback |
| `image_medium` | STRING | Imagen preferida | Render principal |
| `image_large` | STRING | Imagen grande | Render fallback |

---

## 5. BigQuery — `claro_searchai_logs.historial_preguntas`
**Uso principal:** auditoría operativa de preguntas y respuestas emitidas.

| Campo | Tipo (referencial) | Descripción funcional | Uso en API |
|---|---|---|---|
| `pregunta` | STRING | Consulta normalizada | Analítica y troubleshooting |
| `sia_id` | STRING | Identificador de sesión/usuario | Trazabilidad de cliente |
| `pregunta_timestamp` | TIMESTAMP/STRING | Momento original de consulta | Secuencia temporal |
| `timestamp` | TIMESTAMP/STRING | Momento de persistencia | Auditoría técnica |
| `user_agent` | STRING | Agente del cliente | Segmentación técnica |
| `sistema_operativo` | STRING | SO inferido desde user-agent | Analítica |
| `respuesta_json` | STRING(JSON) | Payload de respuesta generado | Reproducción de casos |

---

## 6. BigQuery — `claro_searchai_logs.respuestas_definitivas`
**Uso principal:** cache de respuestas generales reutilizables.

| Campo | Tipo (referencial) | Descripción funcional | Uso en API |
|---|---|---|---|
| `pregunta` | STRING | Clave de consulta normalizada | Lookup de cache |
| `respuesta_json` | STRING(JSON) | Respuesta serializada | Reutilización rápida |
| `timestamp` | TIMESTAMP/STRING | Fecha de almacenamiento | Frescura/rotación |

---

## 7. API externa — `TIENDA_API_URL`
**Uso principal:** fuente productiva para búsquedas de celulares en `/prod/query`.

### Entrada esperada
| Parámetro | Tipo | Descripción |
|---|---|---|
| `q` | STRING | Consulta de búsqueda normalizada |

### Salida esperada (referencial)
| Campo | Tipo | Descripción funcional |
|---|---|---|
| `titulo` | STRING | Encabezado comercial |
| `descripcion` | STRING | Texto explicativo/comercial |
| `producto` | ARRAY | Lista de productos resultantes |
| `recomendados` | ARRAY | Recomendaciones complementarias |
| `status` | STRING | Estado de búsqueda |
| `tipo_respuesta` | STRING | Dominio de respuesta (`tienda`) |

---

## 8. API externa — `CLAROVIDEO_MOVIES_API_URL`
**Uso principal:** búsqueda rápida de títulos de entretenimiento en fallback/atajo.

### Entrada esperada
| Campo | Tipo | Descripción |
|---|---|---|
| `query` | STRING | Título o término de entretenimiento |

### Salida esperada (referencial)
| Campo | Tipo | Descripción funcional |
|---|---|---|
| `status` | STRING | Estado (`Found`/`NotFound`) |
| `listado` | ARRAY | Resultados principales |
| `relacionados` | ARRAY | Contenido relacionado (si aplica) |

---

## 9. Discovery Engine (Vertex) — respuesta general
**Uso principal:** responder categoría `general` con estructura JSON controlada.

### Entrada relevante
| Campo | Tipo | Descripción |
|---|---|---|
| `query.text` | STRING | Consulta de usuario |
| `serving_config` | STRING | Motor/índice configurado |
| `answer_generation_spec` | OBJECT | Reglas de generación de respuesta |

### Salida relevante
| Campo | Tipo | Descripción funcional |
|---|---|---|
| `answer.answer_text` | STRING | Respuesta textual (JSON esperado) |
| `answer.citations` | ARRAY | Fuentes/citas de soporte |
| `answer.citations[].uri` | STRING | URL de referencia |
| `answer.citations[].title` | STRING | Título de referencia |
| `answer.citations[].snippet` | STRING | Fragmento contextual |

---

## 10. Campos de salida de la API (diccionario funcional)

| Campo | Tipo | Origen típico | Comentario |
|---|---|---|---|
| `titulo` | STRING | Tienda API / Vertex / armado local | Encabezado principal |
| `descripcion` | STRING | Tienda API / Vertex / armado local | Mensaje principal |
| `producto` | ARRAY | BigQuery productos o Tienda API | Flujo de tienda/celulares |
| `listado` | ARRAY | Vertex / catálogo video / Movies API | General o entretenimiento |
| `recomendados` | ARRAY | Armado local + BQ/Tienda | Cross-sell |
| `relacionados` | ARRAY | Vertex / catálogo video | Contenido adicional |
| `status` | STRING | Todas las fuentes | `Found`, `NotFound`, `Error` |
| `tipo_respuesta` | STRING | Armado local | `tienda`, `general`, etc. |
| `_meta.normalized_query` | STRING | Lógica local | Observabilidad |
| `_meta.category` | STRING | Clasificación | Observabilidad |
| `_meta.cache_hit` | BOOL | Cache BQ | Observabilidad |

---