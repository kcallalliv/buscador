# Anexo 02 — Contrato API Formal y Ejemplos

## 1. Endpoint
- Método: `GET`
- Rutas:
  - `/prod/query`
  - `/dev/query`

## 2. Parámetros de entrada
- `q` (string, requerido): texto de búsqueda.
- `sia_id` (string, opcional): identificador de sesión/usuario para trazabilidad y rate limit.

## 3. Respuesta base (schema lógico)

```json
{
  "titulo": "string",
  "descripcion": "string",
  "status": "Found | NotFound | Not Found | Error",
  "tipo_respuesta": "tienda | general | pelicula",
  "query": "string",
  "_meta": {
    "normalized_query": "string",
    "category": "celulares | tienda_productos | planes | pelicula | general",
    "cache_hit": "boolean"
  }
}
```

## 4. Variantes por tipo

### 4.1 Respuesta tienda/celulares
```json
{
  "titulo": "¡Los Más Vendidos de Samsung en Claro!",
  "descripcion": "Descubre los equipos más populares...",
  "producto": [
    {"id":"...", "nombre":"Galaxy S24", "url":"https://..."}
  ],
  "recomendados": [
    {"nombre":"Planes Postpago", "url":"https://..."}
  ],
  "status": "Found",
  "tipo_respuesta": "tienda",
  "_meta": {"normalized_query":"samsung", "category":"celulares", "cache_hit":false}
}
```

### 4.2 Respuesta película
```json
{
  "status": "Found",
  "query": "spiderman",
  "tipo": "pelicula",
  "titulo": "¡Disfruta «Spider-Man» en Claro video!",
  "descripcion": "Accede a Claro video...",
  "listado": [
    {"nombre":"Spider-Man", "texto":"...", "url":"https://...", "imagen":"https://..."}
  ],
  "relacionados": [
    {"nombre":"Avengers", "texto":"...", "url":"https://...", "imagen":"https://..."}
  ]
}
```

### 4.3 Respuesta general
```json
{
  "titulo": "¿Cómo puedo ayudarte?",
  "descripcion": "Te ayudo con servicios Claro...",
  "listado": [
    {"nombre":"Centro de Ayuda", "texto":"...", "url":"https://..."}
  ],
  "relacionados": [
    {"nombre":"Tienda Claro", "texto":"...", "url":"https://..."}
  ],
  "status": "Found",
  "tipo_respuesta": "general",
  "query": "como pagar mi recibo"
}
```

## 5. Errores

### 5.1 Parámetro faltante
- HTTP `400`
```json
{"error": "El parámetro 'q' es requerido"}
```

### 5.2 Error interno
- HTTP `500`
```json
{
  "titulo": "Error interno del sistema",
  "descripcion": "Ocurrió un inconveniente al procesar tu solicitud...",
  "query": "...",
  "status": "Error",
  "tipo_respuesta": "general",
  "error_detalle": "..."
}
```

## 6. Consideraciones de contrato
1. `status` no está totalmente unificado (`NotFound` y `Not Found` coexisten).
2. Campos `producto/listado` varían por intención.
3. Frontend debe tolerar campos opcionales y priorizar validación defensiva.
4. Se recomienda versionar contrato como `v1` y migrar a esquema JSON validado.