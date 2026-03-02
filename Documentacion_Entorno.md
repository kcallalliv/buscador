# Anexo 05 — Guía de Operación por Entorno, IAM/Secrets y Release

## 1. Entornos

## 1.1 DEV
- Objetivo: pruebas funcionales y ajuste de heurísticas.
- Endpoint principal: `/dev/query`.
- Puede incluir orígenes CORS de prueba.

## 1.2 PROD
- Objetivo: tráfico real estable.
- Endpoint principal: `/prod/query`.
- Configuración estricta de CORS y control de cambios.

## 2. Variables de entorno mínimas
- `VERTEX_PROJECT`
- `VERTEX_LOCATION`
- `GEMINI_MODEL_NAME`
- `CLAROVIDEO_MOVIES_API_URL`
- `TIENDA_API_URL`
- `PORT`

## 3. IAM/Permisos necesarios
La cuenta de servicio de runtime debe tener permisos para:
1. Ejecutar consultas BigQuery y escritura en tablas de logs/cache.
2. Invocar Discovery Engine/Vertex.
3. Acceso de red saliente a APIs externas configuradas.

## 4. Gestión de secretos
- No hardcodear credenciales en código.
- Usar Secret Manager o mecanismo equivalente del entorno.
- Rotación periódica y trazabilidad de cambios.

## 5. Checklist de release
1. Validar variables por entorno.
2. Verificar permisos IAM efectivos.
3. Desplegar versión candidata.
4. Ejecutar smoke tests de categorías clave.
5. Monitorear 15-30 min.
6. Aprobar release o ejecutar rollback.

## 6. Checklist de rollback
1. Revertir a revisión estable previa.
2. Confirmar recuperación de KPIs.
3. Comunicar estado a stakeholders.
4. Abrir ticket de causa raíz.

## 7. Política de cambios
- Cambios en clasificación/ranking deben incluir casos de prueba canónicos.
- Cambios de contrato API deben ser versionados y comunicados a frontend.
- Cambios en prompt de Vertex deben pasar revisión funcional y de riesgo.