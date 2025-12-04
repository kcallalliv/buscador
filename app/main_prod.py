# app/main_prod.py
import datetime
import json
import logging
import re

from flask import Blueprint, request, jsonify
from flask_cors import CORS

from app import limiter
from app.common import (
    origin_check,
    buscar_respuesta_definitiva,
    guardar_en_respuestas_definitivas,
    guardar_pregunta_en_historial,
)
from app.vertex_handler import get_summary_from_vertex

logger = logging.getLogger(__name__)

prod_bp = Blueprint("prod_bp", __name__)
CORS(prod_bp, resources={r"/prod/*": {"origins": [
    "https://www.claro.com.pe",
    "https://test-claro-pe.prod.clarodigital.net",
]}})

@prod_bp.route("/query", methods=["GET"])
@limiter.limit("10 per minute")
def query_prod():
    user_query = request.args.get("q", "")
    sia_id = request.args.get("sia_id", "")
    user_agent = request.headers.get("User-Agent", "")
    pregunta_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # CORS origin check adicional (duro)
    cors_resp = origin_check(
        request,
        allowed=[
            "https://www.claro.com.pe",
            "https://test-claro-pe.prod.clarodigital.net",
        ],
    )
    if cors_resp:
        return cors_resp

    if not user_query:
        return jsonify({"error": "El parámetro 'q' es requerido"}), 400

    user_query = " ".join(user_query.strip().lower().split())

    try:
        # cache
        respuesta_cache = buscar_respuesta_definitiva(user_query)
        if respuesta_cache:
            logger.info(f"[prod] cache hit: {user_query}")
            guardar_pregunta_en_historial(user_query, sia_id, respuesta_cache, pregunta_timestamp, user_agent)
            return jsonify(json.loads(respuesta_cache))

        # vertex
        nueva_respuesta = get_summary_from_vertex(user_query)
        nueva_respuesta["query"] = user_query

        # cachear solo prod/general
        guardar_en_respuestas_definitivas(user_query, nueva_respuesta)
        guardar_pregunta_en_historial(user_query, sia_id, nueva_respuesta, pregunta_timestamp, user_agent)

        return jsonify(nueva_respuesta)

    except Exception as e:
        logger.error(f"[prod] Error procesando '{user_query}': {e}", exc_info=True)
        return jsonify({
            "titulo": "Error inesperado",
            "descripcion": "<p>Ocurrió un error al procesar tu solicitud.</p>",
            "query": user_query,
            "error_detalle": str(e)
        }), 500
