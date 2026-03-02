# Documentación Técnica Detallada — `app/vertex_handler.py`

## 1. Propósito del módulo
El módulo `vertex_handler.py` encapsula la integración con **Google Discovery Engine (Vertex AI Search/Conversational Search)** para responder consultas de propósito general del buscador de Claro, forzando una estructura JSON de salida homogénea para consumo del frontend/API.

Su función central es:
- construir y enviar la solicitud `AnswerQueryRequest`,
- aplicar políticas de robustez (retry y timeout),
- limpiar y parsear la respuesta del modelo,
- aplicar un fallback seguro cuando el modelo no devuelve JSON válido,
- enriquecer la salida con citas provenientes de Discovery Engine.

---

## 2. Dependencias y componentes usados

### 2.1 Dependencias externas
- `google.cloud.discoveryengine_v1` para cliente de búsqueda conversacional.
- `google.api_core.retry` y `google.api_core.exceptions` para política de reintentos.
- `google.api_core.client_options.ClientOptions` para configurar endpoint regional/global.

### 2.2 Dependencias estándar
- `logging` para trazabilidad.
- `json` para parse de la respuesta del modelo.
- `time` para medir latencia de invocación.

---

## 3. Configuración fija del módulo

El módulo define constantes de configuración:
- `PROJECT_ID = "prd-claro-mktg-data-storage"`
- `LOCATION = "global"`
- `ENGINE_ID = "claro-peru_1748977843565"`

Estas constantes determinan el `serving_config` objetivo del motor de Discovery Engine.

---

## 4. Estrategia de resiliencia

Se declara `VERTEX_RETRY_POLICY` con:
- excepciones reintentables:
  - `ServiceUnavailable` (503),
  - `InternalServerError` (500),
  - `DeadlineExceeded` (timeout).
- backoff exponencial:
  - `initial=1.0`,
  - `multiplier=2.0`,
  - `maximum=10.0`,
  - `deadline=60.0` (tiempo total de reintentos).

Esto mitiga fallas transitorias de infraestructura del proveedor sin romper inmediatamente la experiencia de usuario.

---

## 5. Función principal: `get_summary_from_vertex(user_query)`

## 5.1 Entrada
- `user_query` (str): consulta normalizada o original desde el endpoint llamador.

## 5.2 Construcción de cliente
1. Se evalúa si `LOCATION` es `global`.
2. Si no es `global`, se construye `ClientOptions(api_endpoint=...)`.
3. Se instancia `ConversationalSearchServiceClient`.

## 5.3 Construcción de ruta de serving config
Se arma el identificador completo:
`projects/{PROJECT_ID}/locations/{LOCATION}/collections/default_collection/engines/{ENGINE_ID}/servingConfigs/default_serving_config`

Esto fija de forma explícita el motor que responde.

## 5.4 Prompt y reglas de formato
El módulo arma un `prompt_text` muy estricto que obliga al modelo a retornar JSON con estructura:

```json
{
  "titulo": "string",
  "descripcion": "string",
  "listado": [{"nombre":"string","texto":"string","url":"string"}],
  "relacionados": [{"nombre":"string","texto":"string","url":"string"}],
  "status": "Found o Not Found"
}
```

Además incluye políticas de contenido (priorización personas, coherencia entre nombre/url, tratamiento de temas concretos como streaming/música/celulares, restricciones de afirmaciones comerciales, etc.).

## 5.5 Configuración de generación de respuesta
Se configura `AnswerGenerationSpec` con controles como:
- ignorar consultas adversariales,
- ignorar consultas no orientadas a respuesta,
- ignorar bajo contenido relevante,
- incluir prompt preámbulo (`preamble`),
- idioma de respuesta (`es`),
- y citar resultados.

> Nota técnica: esta configuración reduce ruido y mejora la consistencia del formato consumido por capas superiores.

## 5.6 Construcción de request
Se arma `AnswerQueryRequest` con:
- `serving_config`,
- `query` con texto del usuario,
- `session` temporal,
- `answer_generation_spec`.

## 5.7 Ejecución con observabilidad
La llamada `client.answer_query(...)`:
- usa `retry=VERTEX_RETRY_POLICY`,
- define `timeout` explícito,
- captura tiempo de ejecución para logging de latencia.

## 5.8 Manejo de errores
El módulo diferencia:
- errores de timeout/reintentos agotados,
- errores generales inesperados.

En ambos casos, registra log y propaga excepción para que el endpoint superior decida fallback global HTTP/JSON.

---

## 6. Limpieza y parse de respuesta del modelo

Tras recibir `response.answer.answer_text`:
1. Se hace `strip()`.
2. Se limpia posible envoltura markdown:
   - ```json ... ```
   - ``` ... ```
   - prefijo `json`.
3. Se intenta `json.loads(...)`.

Si el parse falla, el módulo no rompe: usa un **JSON amigable por defecto** que mantiene `status: "Found"` y provee rutas útiles (postpago, prepago, tienda, hogar, claro video, ayuda).

---

## 7. Enriquecimiento con citas (`citations`)

El módulo itera `response.answer.citations` y, por cada cita:
- obtiene enlace estructurado o no estructurado,
- extrae `uri`, `title` y `snippet`,
- construye un bloque HTML enlazado,
- lo agrega como entrada adicional en `listado`.

### Resultado
La respuesta final combina:
- JSON base/fallback,
- y “fuentes adicionales” derivadas de Discovery Engine.

Esto permite trazabilidad de contenido sin cambiar contrato principal de respuesta.

---

## 8. Contrato de salida efectivo

La función retorna un `dict` con al menos:
- `titulo`
- `descripcion`
- `listado`
- `relacionados`
- `status`

y potencialmente elementos extra enviados por el modelo.

---

## 9. Integración con el resto del sistema

`app/main_dev.py` y `app/main_prod.py` invocan `get_summary_from_vertex(...)` en el flujo `category == "general"`.

Después de recibir la respuesta:
- le agregan `query` y `tipo_respuesta`.
- opcionalmente la guardan como respuesta definitiva (cache BigQuery).
- aplican fallback a películas si el contenido sugiere “no encontrado”.

---

## 10. Riesgos y puntos de atención

1. **Prompt largo y con muchas reglas**: puede aumentar varianza o costos de tokens.
2. **HTML embebido en `listado.texto`**: obliga a frontend a sanitizar/renderizar correctamente.
3. **Constantes hardcodeadas** (`PROJECT_ID`, `ENGINE_ID`): conviene externalizar a variables de entorno.
4. **Dependencia de formato JSON del LLM**: aunque hay fallback, la calidad del parse depende del cumplimiento del modelo.

---

## 11. Recomendaciones de mejora

1. Parametrizar `PROJECT_ID`, `LOCATION`, `ENGINE_ID` por entorno.
2. Validar respuesta con esquema JSON (pydantic/jsonschema) antes de retornar.
3. Separar prompt en archivo versionado para control de cambios.
4. Incluir métricas por tipo de fallback (parse_fail, timeout, no_citations, etc.).

---

## 12. Resumen operativo

`vertex_handler.py` es la pasarela de IA conversacional general del sistema. Está diseñado con enfoque de **robustez + contrato estable**: incluso ante respuestas malformadas del modelo o errores transitorios, retorna una estructura útil para que la API no falle y mantenga continuidad de servicio.