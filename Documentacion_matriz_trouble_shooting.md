# Anexo 04 — Matriz de Troubleshooting

## 1. Objetivo
Facilitar diagnóstico rápido con enfoque síntoma → causa probable → verificación → acción.

| Síntoma | Causa probable | Verificación | Acción recomendada |
|---|---|---|---|
| HTTP 400 en muchas solicitudes | Falta parámetro `q` desde frontend | Revisar payload/URL de cliente | Corregir integración frontend y agregar validación temprana |
| HTTP 403 | Origin no permitido | Revisar header `Origin` y lista allowlist | Actualizar allowlist según entorno |
| HTTP 429 frecuentes | Rate limit por `sia_id` o IP | Revisar patrón de tráfico por cliente | Ajustar límite o controlar bursts en cliente |
| HTTP 500 generalizado | Falla en dependencia crítica (BQ/Vertex) | Revisar logs de excepción y latencia dependencia | Rollback o degradar funcionalidad dependiente |
| Solo falla categoría `general` | Discovery Engine timeout/error | Revisar tiempo y errores en `get_summary_from_vertex` | Ajustar timeout/retry, validar IAM/serving config |
| Solo falla categoría `celulares` en PROD | Tienda API inaccesible | Probar URL `TIENDA_API_URL` y respuesta | Restaurar servicio externo o fallback temporal |
| Respuestas vacías/no útiles | JSON del modelo no parseable | Revisar logs de `vertex_answer_text` | Afinar prompt y activar validación de esquema |
| Muchas respuestas `Not Found` en películas | Catálogo no encuentra coincidencias | Validar query BQ y cobertura de títulos | Ajustar scoring y reglas de normalización |
| Latencia alta sostenida | Queries BQ costosas o dependencia lenta | Comparar tiempos por etapa | Optimizar SQL, índices lógicos, límites y caché |
| Duplicidad/inconsistencia dev/prod | Lógica divergente entre `main_dev.py` y `main_prod.py` | Comparar outputs por casos canónicos | Consolidar módulos compartidos |

## 2. Checklist de diagnóstico en 10 minutos
1. Confirmar alcance (global o por categoría).
2. Ejecutar smoke tests de 4 intenciones.
3. Revisar 5xx/latencia por endpoint.
4. Identificar dependencia lenta/fallida.
5. Aplicar mitigación (rollback/degradación).