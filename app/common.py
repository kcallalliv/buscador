# app/common.py
import datetime
import json
import re
from typing import Optional, Tuple, List

from flask import jsonify, Request
from google.cloud import bigquery

# Tablas & BQ client compartidos
bq_client = bigquery.Client()
TABLA_HISTORIAL   = "prd-claro-mktg-data-storage.claro_searchai_logs.historial_preguntas"
TABLA_DEFINITIVAS = "prd-claro-mktg-data-storage.claro_searchai_logs.respuestas_definitivas"

# ----------------------------------------------------------------------------- 
# Utilidades comunes
# -----------------------------------------------------------------------------
def extraer_sistema_operativo(user_agent: Optional[str]) -> str:
    if not user_agent:
        return "Desconocido"
    patrones: List[Tuple[str, str]] = [
        (r'Windows NT 10', 'Windows 10'),
        (r'Windows NT 6.1', 'Windows 7'),
        (r'Android ([\d\.]+)', 'Android'),
        (r'Mac OS X ([\d_\.]+)', 'Mac OS X'),
        (r'iPhone; CPU iPhone OS ([\d_\.]+)', 'iOS'),
        (r'Linux', 'Linux'),
    ]
    for patron, nombre in patrones:
        if re.search(patron, user_agent):
            return nombre
    return "Otro"

def origin_check(request: Request, *, allowed: Optional[list] = None):
    """
    Valida el header Origin contra una lista permitida.
    - Si no hay Origin (p. ej. prueba directa en navegador), no bloquea.
    - Si hay Origin y no está permitido, devuelve (json, 403).
    - Si todo ok, devuelve None.
    """
    if not allowed:
        return None
    origin = request.headers.get("Origin")
    if origin and origin not in allowed:
        return jsonify({"error": "Acceso no autorizado"}), 403
    return None

# ----------------------------------------------------------------------------- 
# Cache / Historial
# -----------------------------------------------------------------------------
def buscar_respuesta_definitiva(pregunta_normalizada: str) -> Optional[str]:
    query = f"""
        SELECT respuesta_json
        FROM `{TABLA_DEFINITIVAS}`
        WHERE pregunta = @pregunta
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("pregunta", "STRING", pregunta_normalizada)]
    )
    resultados = bq_client.query(query, job_config=job_config).result()
    filas = list(resultados)
    return filas[0]["respuesta_json"] if filas else None

def guardar_en_respuestas_definitivas(pregunta_normalizada: str, respuesta_json):
    # SOLO debe usarse desde "general"
    if isinstance(respuesta_json, dict):
        respuesta_json_str = json.dumps(respuesta_json, ensure_ascii=False)
    elif isinstance(respuesta_json, str):
        json.loads(respuesta_json)  # valida que sea JSON
        respuesta_json_str = respuesta_json
    else:
        return

    fila = [{
        "pregunta": pregunta_normalizada,
        "respuesta_json": respuesta_json_str,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }]
    bq_client.insert_rows_json(TABLA_DEFINITIVAS, fila)

def guardar_pregunta_en_historial(
    pregunta_normalizada: str,
    sia_id: str,
    respuesta_json,
    pregunta_timestamp: str,
    user_agent: Optional[str],
    so: Optional[str] = None,
):
    """
    Almacena el historial de consultas. `so` es opcional; si no viene lo inferimos del user_agent.
    """
    if isinstance(respuesta_json, dict):
        respuesta_json_str = json.dumps(respuesta_json, ensure_ascii=False)
    elif isinstance(respuesta_json, str):
        json.loads(respuesta_json)
        respuesta_json_str = respuesta_json
    else:
        return

    so = so or extraer_sistema_operativo(user_agent)
    fila = [{
        "pregunta": pregunta_normalizada,
        "sia_id": sia_id,
        "pregunta_timestamp": pregunta_timestamp,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "user_agent": user_agent,
        "sistema_operativo": so,
        "respuesta_json": respuesta_json_str
    }]
    bq_client.insert_rows_json(TABLA_HISTORIAL, fila)
