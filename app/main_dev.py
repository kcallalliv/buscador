# app/main_dev.py
import requests
import datetime
import json
import logging
import os
import re
import hashlib
import unicodedata
from difflib import SequenceMatcher
from typing import List, Tuple, Optional
from rapidfuzz import fuzz

from flask import Blueprint, request, jsonify
from flask_cors import CORS
from google.cloud import bigquery
from vertexai import init as vertexai_init
from vertexai.generative_models import GenerativeModel, GenerationConfig

from app import limiter
from app.common import (
    origin_check,
    buscar_respuesta_definitiva,
    guardar_en_respuestas_definitivas,
    guardar_pregunta_en_historial,
)
from app.vertex_handler import get_summary_from_vertex

logger = logging.getLogger(__name__)

# ------------------ Config Vertex / Gemini ------------------
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "prd-claro-mktg-data-storage")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.0-flash")

vertexai_init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
_gemini_model = GenerativeModel(GEMINI_MODEL_NAME)
_gemini_genconf = GenerationConfig(
    temperature=0.2,
    top_p=0.9,
    top_k=32,
    response_mime_type="application/json"
)

CLAROVIDEO_MOVIES_API_URL = os.environ.get(
    "CLAROVIDEO_MOVIES_API_URL",
    "https://clarovideo-movies-api-1079186964678.us-central1.run.app/v1/clarovideo/peliculas/search"
)

TELCO_BLOCK_WORDS = [
    "chip","portabilidad","recarga","plan","planes","roaming","internet","datos","ilimitado",
    "factura","deuda","pago","pagos","linea","línea","numero","número",
    "hogar","fibra","wifi","router","modem","módem","instalacion","instalación",
    "tv","cable","claro tv","atencion","atención","soporte","reclamo","reclamos",
    "servicio","servicios","migrar","renovar","renovacion","renovación"
]

DEVICE_BLOCK = {
    "iphone","apple","samsung","xiaomi","huawei","motorola","oppo","vivo",
    "redmi","realme","honor","nokia","sony","galaxy","pixel"
}

def looks_like_title(q: str) -> bool:
    ql = (q or "").strip().lower()
    if len(ql) < 2 or len(ql) > 60:
        return False
    if " " in ql and len(ql.split()) > 8:
        return False
    if any(w in ql for w in TELCO_BLOCK_WORDS):
        return False
    if ql in DEVICE_BLOCK or any(w in ql for w in DEVICE_BLOCK):
        return False
    return True


def try_movies_first(normalized_query: str) -> dict | None:
    if not looks_like_title(normalized_query):
        return None

    try:
        r = requests.post(
            CLAROVIDEO_MOVIES_API_URL,
            json={"query": normalized_query},
            timeout=3
        )
        r.raise_for_status()
        data = r.json()

        if data.get("status") != "Found" or not data.get("listado"):
            return None

        return data
    except Exception:
        return None

# ------------------ BQ ------------------
bq_client = bigquery.Client()
BQ_TABLE_PRODUCTS = "prd-claro-mktg-data-storage.tienda_claro.productos"
BQ_TABLE_TOP5 = "prd-claro-mktg-data-storage.master_analytics.sales_top5_brand_monthly"
BQ_TABLE_CLAROVIDEO = "prd-claro-mktg-data-storage.claro_video.catalogo" 

# ------------------ Blueprint ------------------
dev_bp = Blueprint("dev_bp", __name__)
#CORS(dev_bp, resources={r"/dev/*": {"origins": [
#    "https://www.claro.com.pe",
#    "https://test-claro-pe.prod.clarodigital.net",
#    "https://search-test-1079186964678.us-central1.run.app",
#    "https://search-api-1079186964678.us-central1.run.app",
#    "https://genia-front-test-1079186964678.us-central1.run.app",
#]}})

DEV_ALLOWED_ORIGINS = [
    "https://www.claro.com.pe",
    "https://test-claro-pe.prod.clarodigital.net",
    "https://search-test-1079186964678.us-central1.run.app",
    "https://search-api-1079186964678.us-central1.run.app",
    "https://genia-front-test-1079186964678.us-central1.run.app"]
CORS(dev_bp, resources={r"/dev/*": {"origins": DEV_ALLOWED_ORIGINS}})

def call_clarovideo_movies_api(normalized_query: str) -> dict:
    payload = {"query": normalized_query}
    r = requests.post(
        CLAROVIDEO_MOVIES_API_URL,
        json=payload,
        timeout=10,
        headers={"Content-Type": "application/json"},
    )
    r.raise_for_status()
    return r.json()


# ------------------ Helpers Vertex ------------------

def title_is_clarovideo_disfruta(respuesta: dict) -> bool:
    titulo = (respuesta.get("titulo") or "").strip().lower()
    return titulo.startswith("claro video: disfruta de")

def description_is_clarovideo_disfruta(respuesta: dict) -> bool:
    titulo = (respuesta.get("titulo") or "").lower()
    return "y mucho más con claro video" in titulo

def description_is_clarovideo_genero(respuesta: dict) -> bool:
    titulo = (respuesta.get("titulo") or "").lower()
    return "películas " in titulo


def call_gemini_json(*, system_prompt: str, user_input: str, timeout: int = 15) -> dict:
    resp = _gemini_model.generate_content(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        generation_config=_gemini_genconf,
        timeout=timeout,
    )
    text = (resp.text or "").strip()
    if not text:
        return {"normalized_query": user_input, "category": "general"}
    try:
        return json.loads(text)
    except Exception:
        return {"normalized_query": user_input, "category": "general"}

# ------------------ Normalización / Clasificación ------------------
ROMAN_MAP = {
    "xiv": "14", "xv": "15", "xvi": "16", "xvii": "17", "xviii": "18", "xix": "19", "xx": "20"
}

def roman_to_arabic(s: str) -> str:
    out = s
    for r, a in ROMAN_MAP.items():
        out = re.sub(rf"\b{r}\b", a, out)
    return out

def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def sanitize_query(q: str) -> str:
    if q is None:
        return ""
    q = unicodedata.normalize("NFKC", q)
    q = q.replace("\u00A0", " ")
    q = re.sub(r"[\u0000-\u001F\u007F]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

WORD_TYPO_MAP = {
    "ifon": "iphone", "iphon": "iphone", "ipone": "iphone", "aifon": "iphone",
    "galaxi": "galaxy", "sansumg": "samsung", "samgsung": "samsung", "samung": "samsung", "samsumg":"samsung",
    "xiomi": "xiaomi", "huawe": "huawei", "huauei": "huawei", "motrola": "motorola", "redmy": "redmi",
    "motog": "moto g", "audifono": "audifonos", "audífono": "audifonos", "audífonos": "audifonos",
    "inalambrico": "inalambricos", "inalámbrico": "inalambricos", "aipods":"airpods", "eardpos":"earpods",
    "xiamy":"xiaomi", "xiami":"xiaomi", "gaxaliny":"galaxy", "gaxaly":"galaxy", "samsug":"samsung"
}
PHRASE_TYPO_MAP = {"router claro": "router"}

def _fix_special_typos(s: str) -> str:
    return re.sub(r"\baudiphone+o+s?\b", "audifonos", s)

def normalize_query_local(q: str) -> str:
    s = sanitize_query(q)
    s = strip_accents(s.lower()).strip()
    s = roman_to_arabic(s)
    s = re.sub(r"\s+", " ", s)
    for k, v in PHRASE_TYPO_MAP.items():
        s = s.replace(k, v)
    for k, v in WORD_TYPO_MAP.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s)
    s = _fix_special_typos(s)
    return s

def _safe_pick_normalized(model_norm: str, local_norm: str) -> str:
    b = strip_accents((model_norm or "").lower()).strip()
    a = local_norm
    if not b:
        return a
    ratio = SequenceMatcher(None, a, b).ratio()
    if ratio < 0.6 or abs(len(a) - len(b)) > max(5, int(len(a) * 0.3)):
        return a
    return b

BRAND_HINTS = {
    "apple","iphone","samsung","galaxy","xiaomi","redmi","poco","huawei","honor",
    "motorola","moto","nokia","google","pixel","oneplus","oppo","realme","vivo",
    "zte","tcl","alcatel","sony","asus","lenovo","infinix","tecno","nothing","meizu","umidigi","doogee","blackview","blu","fairphone","cat"
}

OTHER_BRAND_TERMS = {
    "apple":   ["galaxy","samsung","xiaomi","redmi","poco","motorola","huawei","honor","oppo","realme","vivo","pixel","google","tecno","infinix","lenovo","zte","tcl","nokia"],
    "samsung": ["iphone","apple","xiaomi","redmi","poco","motorola","huawei","honor","oppo","realme","vivo","pixel","google","tecno","infinix","lenovo","zte","tcl","nokia"],
    "xiaomi":  ["iphone","apple","galaxy","samsung","motorola","huawei","honor","oppo","realme","vivo","pixel","google","tecno","infinix","lenovo","zte","tcl","nokia"],
    "motorola":["iphone","apple","galaxy","samsung","xiaomi","redmi","poco","huawei","honor","oppo","realme","vivo","pixel","google","tecno","infinix","lenovo","zte","tcl","nokia"],
    "huawei":  ["iphone","apple","galaxy","samsung","xiaomi","redmi","poco","motorola","honor","oppo","realme","vivo","pixel","google","tecno","infinix","lenovo","zte","tcl","nokia"],
    "honor":   ["iphone","apple","galaxy","samsung","xiaomi","redmi","poco","motorola","huawei","oppo","realme","vivo","pixel","google","tecno","infinix","lenovo","zte","tcl","nokia"],
    "google":  ["iphone","apple","galaxy","samsung","xiaomi","redmi","poco","motorola","huawei","honor","oppo","realme","vivo","tecno","infinix","lenovo","zte","tcl","nokia"],
    "oppo":    ["iphone","apple","galaxy","samsung","xiaomi","redmi","poco","motorola","huawei","honor","realme","vivo","pixel","google","tecno","infinix","lenovo","zte","tcl","nokia"],
    "realme":  ["iphone","apple","galaxy","samsung","xiaomi","redmi","poco","motorola","huawei","honor","oppo","vivo","pixel","google","tecno","infinix","lenovo","zte","tcl","nokia"],
    "vivo":    ["iphone","apple","galaxy","samsung","xiaomi","redmi","poco","motorola","huawei","honor","oppo","realme","pixel","google","tecno","infinix","lenovo","zte","tcl","nokia"],
    "lenovo":  ["iphone","apple","galaxy","samsung","xiaomi","redmi","poco","motorola","huawei","honor","oppo","realme","vivo","pixel","google","tecno","infinix","zte","tcl","nokia"],
    "zte":     ["iphone","apple","galaxy","samsung","xiaomi","redmi","poco","motorola","huawei","honor","oppo","realme","vivo","pixel","google","tecno","infinix","lenovo","tcl","nokia"],
}

ACCESSORY_TERMS = {
    "power bank","powerbank","bateria externa","batería externa","cargador","cargadores","cable",
    "audifono","audifonos","earbuds","handsfree","parlante","speaker","earpods","earpod","airpods",
    "smartwatch","reloj inteligente","band","pulsera","wearable","protector","funda","case",
    "memoria","microsd","micro sd","sd card","ipad","tablet","apple pencil","cargador inalambrico","cargador inalámbrico",
    "auriculares"
}
STORE_SIGNALS = {
    "router","modem","módem","decodificador","deco","stb","set top box","set-top box","ont","gpon",
    "mesh","repetidor","extensor","wifi 6","wifi6","fibra","hfc","internet hogar","internet fijo",
    "telefonia fija","telefónica fija","linea fija","línea fija",
    "claro tv","television","televisión","tv",
    "1 play","2 play","3 play","one play","two play","triple play","doble play",
    "plan hogar","plan negocios","negocios","empresas","fijos","telefonia 5000","telefonía 5000",
}

PHONE_MODEL_PATTERNS = [
    re.compile(r"\biphone\s?(?:se|[0-9]{1,2})(?:\s?(?:pro\s?max|pro|plus|mini|ultra))?\b"),
    re.compile(r"\bi\s*phone\b"),
    re.compile(r"\bgalaxy\s?(?:s|a|m|z)\s?-?\s?(?:flip|fold)?\s?[0-9]{1,3}[a-z]?(?:\s?(?:fe|ultra|plus))?\b"),
    re.compile(r"\b(?:s|a|m)[0-9]{1,3}\b"),
    re.compile(r"\bz\s?(?:flip|fold)\s?[0-9]?\b"),
    re.compile(r"\bnote\s?[0-9]{1,2}\b"),
    re.compile(r"\bredmi\s?(?:note|[a-z])\s?[0-9]{1,3}[a-z]?(?:\s?(?:pro|plus|prime|turbo|ultra))?\b"),
    re.compile(r"\bpoco\s?(?:f|x|m)\s?[0-9]{1,3}[a-z]?(?:\s?(?:pro|gt|ultra))?\b"),
    re.compile(r"\bxiaomi\s?(?:[0-9]{1,3}[a-z]?|mi\s?[0-9]{1,3})(?:\s?(?:t|pro|lite|ultra))?\b"),
    re.compile(r"\bmoto\s?(?:g|e|edge)\s?[0-9]{1,3}[a-z]?(?:\s?(?:plus|pro|power|fusion|neo|ultra))?\b"),
    re.compile(r"\bhuawei\s?(?:p|mate|nova)\s?[0-9]{1,3}[a-z]?(?:\s?(?:pro|lite|ultra))?\b"),
    re.compile(r"\bhonor\s?(?:[0-9]{1,3}|magic|x|play)\s?[0-9]{0,3}[a-z]?(?:\s?(?:pro|lite|ultra))?\b"),
    re.compile(r"\bpixel\s?[0-9]{1,2}(?:\s?(?:a|pro|xl))?\b"),
    re.compile(r"\bone\s?plus\s?[0-9]{1,2}(?:\s?(?:t|r|pro))?\b"),
    re.compile(r"\boppo\s?(?:reno|find|a)\s?[0-9]{1,3}[a-z]?(?:\s?(?:pro|plus|x|n))?\b"),
    re.compile(r"\brealme\s?(?:c|[0-9])\s?[0-9]{1,3}[a-z]?(?:\s?(?:pro|plus|narzo|ultra))?\b"),
    re.compile(r"\bvivo\s?(?:y|v|t|x)\s?[0-9]{1,3}[a-z]?(?:\s?(?:pro|plus))?\b"),
    re.compile(r"\binfinix\s?(?:hot|note|zero|smart)\s?[0-9]{1,3}[a-z]?\b"),
    re.compile(r"\btecno\s?(?:spark|camon|pova|phantom)\s?[0-9]{1,3}[a-z]?\b"),
    re.compile(r"\bnokia\s?(?:c|g|x)\s?[0-9]{1,3}[a-z]?\b"),
    re.compile(r"\bzte\s?(?:blade|axon)\s?\w*\b"),
    re.compile(r"\btcl\s?[0-9a-z]+\b"),
    re.compile(r"\balcatel\s?\w+\b"),
    re.compile(r"\bxperia\s?\w+\b"),
    re.compile(r"\brog\s?phone\s?[0-9]?\b"),
    re.compile(r"\blenovo\s?(?:k|z)\s?[0-9]{1,3}[a-z]?\b"),
    re.compile(r"\bnothing\s?phone\s?\(?[12](?:a|\.?5)?\)?\b"),
    re.compile(r"\bfairphone\s?[0-9]\b"),
    re.compile(r"\bcat\s?s[0-9]{2}\b"),
    re.compile(r"\bblackview\s?\w+\b"),
    re.compile(r"\bdoogee\s?\w+\b"),
    re.compile(r"\bumidigi\s?\w+\b"),
    re.compile(r"\bblu\s?\w+\b"),
    re.compile(r"\bmeizu\s?\w+\b"),
]

MOVIE_HINTS = {
    "spiderman","spider-man","avengers","iron man","captain america","thor","hulk","deadpool",
    "batman","joker","superman","flash","aquaman","wonder woman","suicide squad",
    "star wars","harry potter","lord of the rings","el señor de los anillos","jurassic",
    "transformers","fast and furious","rapidos y furiosos","furious","toy story","frozen",
    "encanto","minions","despicable me","megamind","inside out","intensamente",
    "dune","matrix","avatar","oppenheimer","barbie","godzilla","kong","shrek",
    "mario","sonic","mission impossible","john wick","predator","alien","terminator"
}

PLAN_HINTS = {
    "plan", "planes", "postpago", "prepago", "pospago", "linea nueva", "línea nueva", "portabilidad", "renovacion", "renovación",
    "chip", "sim card", "tarjeta sim"
}

def _match_any(patterns, text: str) -> bool:
    return any(p.search(text) for p in patterns)

# --- 1. OPTIMIZACIÓN DE CLARO VIDEO ---
def search_claro_video_catalog(
    normalized_query: str,
    max_main: int = 4,
    max_related: int = 5,
):
    nq = normalize_query_local(normalized_query)
    nq_base = strip_accents((nq or "").lower()).strip()
    if not nq_base:
        return [], []

    tokens = [re.escape(t) for t in nq_base.split() if len(t) > 2]
    if not tokens:
         regex_filter = r".+" 
    else:
        regex_filter = "|".join(tokens)

    sql = f"""
    SELECT
      id, section, title, title_original, year, duration,
      description, description_large, rating_code, title_uri,
      url, image_small, image_medium, image_large
    FROM `{BQ_TABLE_CLAROVIDEO}`
    WHERE REGEXP_CONTAINS(LOWER(CONCAT(IFNULL(title,''), ' ', IFNULL(title_original,''))), @regex_filter)
    LIMIT 200
    """

    try:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("regex_filter", "STRING", regex_filter),
            ]
        )
        rows = list(bq_client.query(sql, job_config=job_config).result())
    except Exception as e:
        logger.exception("Error consultando catálogo Claro video", exc_info=e)
        return [], []

    if not rows:
        return [], []

    def row_to_movie(r) -> dict:
        return {
            "id": r.id,
            "section": r.section,
            "title": r.title,
            "title_original": r.title_original,
            "year": r.year,
            "duration": r.duration,
            "description": r.description,
            "description_large": r.description_large,
            "rating_code": r.rating_code,
            "title_uri": r.title_uri,
            "url": r.url,
            "image_small": r.image_small,
            "image_medium": r.image_medium,
            "image_large": r.image_large,
        }

    scored = []
    for row in rows:
        title = strip_accents((row.title or "").lower())
        title_orig = strip_accents((row.title_original or "").lower())
        slug = strip_accents(((row.title_uri or "") or "").replace("-", " ").lower())

        s1 = fuzz.WRatio(nq_base, title) if title else 0
        s2 = fuzz.WRatio(nq_base, title_orig) if title_orig else 0
        s3 = fuzz.WRatio(nq_base, slug) if slug else 0

        score = max(s1, s2, s3)
        scored.append((score, row, title, title_orig, slug))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored or scored[0][0] < 40:
        return [], []

    main_movies: List[dict] = []
    related_movies: List[dict] = []

    for score, row, title, title_orig, slug in scored:
        if score < 40: break
        if (nq_base in title or nq_base in title_orig or nq_base in slug):
            main_movies.append(row_to_movie(row))
        if len(main_movies) >= max_main: break

    if not main_movies:
        for score, row, *_ in scored:
            if score < 40: break
            main_movies.append(row_to_movie(row))
            if len(main_movies) >= max_main: break

    used_ids = {m["id"] for m in main_movies}

    for score, row, *_ in scored:
        if len(related_movies) >= max_related: break
        if row.id in used_ids: continue
        if score < 30: continue
        related_movies.append(row_to_movie(row))

    # Backfill para relacionados
    if len(related_movies) < max_related and main_movies:
        try:
            ref_item = main_movies[0]
            ref_section = ref_item.get("section")
            exclude_ids = [m["id"] for m in main_movies + related_movies]
            
            backfill_limit = max_related - len(related_movies)
            
            backfill_sql = f"""
            SELECT
              id, section, title, title_original, year, duration,
              description, description_large, rating_code, title_uri,
              url, image_small, image_medium, image_large
            FROM `{BQ_TABLE_CLAROVIDEO}`
            WHERE id NOT IN UNNEST(@exclude_ids)
            """
            
            query_params = [
                bigquery.ArrayQueryParameter("exclude_ids", "STRING", exclude_ids),
            ]

            if ref_section:
                backfill_sql += " AND section = @ref_section"
                query_params.append(bigquery.ScalarQueryParameter("ref_section", "STRING", ref_section))
            
            backfill_sql += f" LIMIT {backfill_limit + 5}"

            job_config_bf = bigquery.QueryJobConfig(query_parameters=query_params)
            bf_rows = list(bq_client.query(backfill_sql, job_config=job_config_bf).result())
            
            for row in bf_rows:
                if len(related_movies) >= max_related: break
                related_movies.append(row_to_movie(row))
                
        except Exception as e:
            logger.warning(f"Error en backfill de relacionados: {e}")

    return main_movies, related_movies

GEMINI_INTENT_PROMPT = """
Eres un clasificador de consultas para el buscador de Claro Perú.
Tu tarea: (1) corregir/estandarizar la consulta; (2) clasificarla en una categoría.

Categorías:
- "celulares": equipos/marcas/modelos de teléfonos (iphone, samsung galaxy, xiaomi/redmi/poco, motorola, huawei/honor, etc.).
- "tienda_productos": catálogo de Tienda Claro NO celulares (routers, decodificadores, planes hogar/negocios fijos, accesorios como power bank/cargadores/audífonos/smartwatch, etc.).
- "planes": intención de planes móviles (postpago, prepago, portabilidad, renovación) o chip.
- "pelicula": intención de entretenimiento en Claro video: título de película O serie, franquicia/saga, personaje/animación("Bob Esponja", "Dragon Ball"), género/categoría ("películas de terror", "películas románticas"), o búsquedas típicas para ver contenido o pregunten por animales o sobre batallas.
  Si la consulta parece de entretenimiento/streaming, responde "pelicula" aunque no sea un título exacto.
- "general": Preguntas de soporte, trámites, preguntas informativas u otros no cubiertos como palabras inentendibles.

Reglas:
- Si la consulta contiene "pelicula de" o "peliculas de", clasifica como "pelicula" y normaliza quitando ese prefijo.
- Si la consulta es marca/modelo de celular o términos telco (chip, plan, portabilidad, fibra, hogar, router, internet, recibo, deuda, migrar a claro), NO es "pelicula".

Ejemplos (solo guía):
- "peliculas de terror" -> {"normalized_query":"terror","category":"pelicula"}
- "bob esponja" -> {"normalized_query":"bob esponja","category":"pelicula"}
- "orgullo y prejuicio" -> {"normalized_query":"orgullo y prejuicio","category":"pelicula"}
- "iphone 15 pro" -> {"normalized_query":"iphone 15 pro","category":"celulares"}
- "plan postpago" -> {"normalized_query":"plan postpago","category":"planes"}

Responde SOLO JSON:
{ "normalized_query": "<minúsculas y corregida>", "category": "celulares" | "tienda_productos" | "planes" | "pelicula" | "general" }
"""


# --- 2. OPTIMIZACIÓN CLASIFICACIÓN (Fast-Path) ---
def classify_with_gemini(user_query: str) -> Tuple[str, str]:
    nq_local = normalize_query_local(user_query)
    
    # Fast-Path
    has_phone_model = _match_any(PHONE_MODEL_PATTERNS, nq_local)
    if has_phone_model:
        return nq_local, "celulares"

    terms = set(nq_local.split())
    if bool(terms & PLAN_HINTS) and not bool(terms & BRAND_HINTS):
        return nq_local, "planes"
        
    if any(kw in nq_local for kw in STORE_SIGNALS) or any(kw in nq_local for kw in ACCESSORY_TERMS):
         return nq_local, "tienda_productos"

    try:
        payload = {"query": user_query}
        result = call_gemini_json(system_prompt=GEMINI_INTENT_PROMPT, user_input=json.dumps(payload), timeout=4)
        model_norm = normalize_query_local(result.get("normalized_query") or nq_local)
        normalized = _safe_pick_normalized(model_norm, nq_local)
        category = (result.get("category") or "general").strip().lower()
    except Exception:
        normalized = nq_local
        category = "general"

    # Overrides finales
    has_phone_model = _match_any(PHONE_MODEL_PATTERNS, normalized)
    terms = set(normalized.split())
    has_phone_brand = bool(terms & BRAND_HINTS)
    txt = f" {normalized} "
    has_accessory = any(f" {kw} " in txt for kw in ACCESSORY_TERMS) or ('airpods' in normalized) or ('earpods' in normalized)
    has_store = any(f" {kw} " in txt for kw in STORE_SIGNALS)
    generic_phone_words = any(w in normalized for w in ["celular","smartphone","equipo","postpago","prepago","liberado","libre"])
    is_movie = any(h in normalized for h in MOVIE_HINTS)
    is_plan_query = bool(terms & PLAN_HINTS) and not (has_phone_model or has_phone_brand or has_accessory)
    
    if is_movie: category = "pelicula"
    elif is_plan_query and not has_phone_model: category = "planes"
    elif has_phone_model: category = "celulares"
    elif has_store or has_accessory: category = "tienda_productos"
    elif has_phone_brand and generic_phone_words: category = "celulares"
    elif has_phone_brand: category = "celulares"
    elif is_plan_query: category = "planes"

    return normalized, category

# ------------------ Utilities ------------------
def _terms_from_query(nq: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", nq)
    seen, out = set(), []
    for t in toks:
        if len(t) >= 2 and t not in seen:
            seen.add(t)
            out.append(t)
    return toks[:6]

def _nums_from_query(nq: str) -> List[str]:
    return re.findall(r"\b\d{1,4}\b", nq)

FILLER_QUERY_TERMS = {
    "quiero", "comprar", "compro", "compra", "comprarlo", "comprarla", "adquirir",
    "adquirirme", "busco", "buscar", "deseo", "necesito", "quisiera", "quieres",
    "dame", "muestrame", "muestra", "mostrar", "ver", "cotizar", "precio", "precios",
    "oferta", "ofertas", "promo", "promocion", "promociones", "nuevo", "nueva",
    "nuevos", "nuevas", "un", "una", "el", "la", "los", "las", "de", "del", "para",
    "por", "favor", "porfa", "mi", "tu", "su", "equipo", "celular", "smartphone",
    "que", "qué", "recomiendas", "tienes", "disponible", "disponibles", "sugerir", "sugieres", "recomendacion","regalar", "no","se","que"
}

COLOR_PATTERNS = {
    "negro": r"(negro|black)",
    "blanco": r"(blanco|white)",
    "azul": r"(azul|blue)",
    "verde": r"(verde|green)",
    "rojo": r"(rojo|red)",
    "rosado": r"(rosado|rosa|pink)",
    "morado": r"(morado|violeta|purple)",
    "gris": r"(gris|gray|grey)",
    "plata": r"(plata|silver)",
    "dorado": r"(dorado|gold)",
    "titanio": r"(titanio|titanium)",
}

BRAND_TERMS_FOR_MODEL = {
    "iphone","samsung","galaxy","xiaomi","redmi","poco","huawei",
    "motorola","moto","pixel","honor","infinix","tecno","oppo","realme","vivo","lenovo","zte"
}

def _brand_terms_from_query(nq: str) -> List[str]:
    terms = set(_terms_from_query(nq))
    return sorted(list(terms & BRAND_TERMS_FOR_MODEL))

def _canonical_brand_from_query(nq: str) -> Optional[str]:
    s = nq.lower()
    if "iphone" in s or "apple" in s: return "apple"
    if "samsung" in s or "galaxy" in s: return "samsung"
    if "xiaomi" in s or "redmi" in s or re.search(r"\bpoco\b", s): return "xiaomi"
    if "huawei" in s: return "huawei"
    if "motorola" in s or re.search(r"\bmoto\b", s): return "motorola"
    if "pixel" in s or "google" in s: return "google"
    if "honor" in s: return "honor"
    if "infinix" in s: return "infinix"
    if "tecno" in s: return "tecno"
    if "oppo" in s: return "oppo"
    if "realme" in s: return "realme"
    if "vivo" in s: return "vivo"
    if "lenovo" in s: return "lenovo"
    if "zte" in s: return "zte"
    return None

def _strip_fillers_from_query(nq: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", nq.lower())
    cleaned = [t for t in tokens if t not in FILLER_QUERY_TERMS]
    if not cleaned:
        return nq
    return " ".join(cleaned)

def _color_from_query(nq: str) -> Tuple[str, str]:
    for key, pattern in COLOR_PATTERNS.items():
        if re.search(rf"\b{key}\b", nq):
            return key, pattern
        if re.search(pattern, nq):
            return key, pattern
    return "", ""

def _extract_color_from_title(title: str) -> str:
    t = strip_accents((title or "").lower())
    for key, pattern in COLOR_PATTERNS.items():
        if re.search(pattern, t):
            return key
    return ""

def _extract_storage_from_title(title: str) -> int:
    t = strip_accents((title or "").lower())
    m = re.search(r"\b(\d{2,4})\s*gb\b", t)
    return int(m.group(1)) if m else 0

def _extract_model_number_from_title(title: str, brand: str) -> int:
    t = strip_accents((title or "").lower())
    t = re.sub(r"\b\d{2,4}\s*gb\b", "", t)
    numbers = [int(n) for n in re.findall(r"\b(\d{1,2})\b", t)]
    if brand == "apple":
        candidates = [n for n in numbers if 10 <= n <= 30]
    else:
        candidates = [n for n in numbers if 1 <= n <= 99]
    return max(candidates) if candidates else 0

def _variant_rank(title: str) -> int:
    t = strip_accents((title or "").lower())
    if re.search(r"\bpro[\s-]*max\b|\bpromax\b", t):
        return 3
    if re.search(r"\bpro\b", t):
        return 2
    if re.search(r"\bplus\b|\b\+\b", t):
        return 1
    return 0

def _brand_display_label(nq: str) -> Optional[str]:
    s = nq.lower()
    if "iphone" in s: return "iPhone"
    if "galaxy" in s: return "Galaxy"
    if "redmi" in s: return "Redmi"
    if "poco" in s: return "Poco"
    if "pixel" in s: return "Pixel"
    if "moto" in s or "motorola" in s: return "Motorola"
    canon = _canonical_brand_from_query(s)
    if not canon:
        return None
    if canon == "xiaomi": return "Xiaomi"
    if canon == "samsung": return "Samsung"
    if canon == "apple": return "iPhone"
    return canon.capitalize()

def _format_phone_label(label: str) -> str:
    if not label:
        return label
    words = []
    for token in label.split():
        if token.lower() == "iphone":
            words.append("iPhone")
        elif token.lower() == "galaxy":
            words.append("Galaxy")
        else:
            words.append(token.title())
    return " ".join(words)

def _display_query_for_phone(nq: str) -> str:
    cleaned = _strip_fillers_from_query(nq)
    label = _brand_display_label(cleaned)
    if label and not _match_any(PHONE_MODEL_PATTERNS, cleaned) and not _nums_from_query(cleaned):
        return label
    return _format_phone_label(cleaned)

BRAND_RESPONSE_TEMPLATES = [
    "¡Excelente elección! Elegir un {brand} es apostar por innovación, seguridad y rendimiento que se renueva cada año. Mira estos modelos disponibles y encuentra el que mejor se adapta a ti 👇",
    "El {brand} es sinónimo de tecnología premium. Cada generación mejora en cámara, potencia y experiencia de uso. Aquí tienes los modelos más destacados para que elijas el ideal 📱✨",
    "Buena decisión, estás eligiendo uno de los smartphones más completos del mercado. Déjame mostrarte las mejores opciones de {brand} disponibles ahora mismo 👇",
    "Qué gran oportunidad de tener un {brand}. Diseño elegante, rendimiento potente y un ecosistema que lo hace todo más fácil. Descubre estos modelos y elige el que va contigo 🚀",
    "Un {brand} no es solo un celular, es una experiencia. Fotos increíbles, fluidez total y actualizaciones constantes. Mira estos modelos disponibles y da el siguiente paso 📲",
    "Si estás pensando en un {brand}, vas por el camino correcto. Estos modelos destacan por su potencia, cámara y estilo. Elige el tuyo y lúcete 🔥",
    "Sabemos que buscas calidad y tecnología de primer nivel. Por eso, el {brand} es una gran elección. Aquí tienes opciones que se ajustan a distintos planes y necesidades 👇",
    "Comprar un {brand} es invertir en durabilidad y confianza. Rendimiento estable hoy y por muchos años más. Revisa estos modelos y encuentra el que mejor encaja contigo ✅",
    "Es el momento perfecto para dar el salto a un {brand}. Innovación, potencia y elegancia en un solo equipo. Mira las opciones disponibles y elige el tuyo ahora ⏳📱",
    "¡Buenísima elección! El {brand} sigue siendo uno de los celulares más completos del mercado. Te dejo aquí los modelos disponibles para que compares y elijas el ideal para ti 👇",
]

PHONE_TITLE_EXACT_TEMPLATES = [
    "¡Qué buena elección! Encontramos el {label}. ¿Cuál te enamora hoy? 😍",
    "¡Listo! El {label} está aquí. Elige tu favorito y sigue 👇",
    "Te escuché: buscas el {label}. Compáralos y elige el tuyo ✅",
    "¡Vamos con el {label}! Mira opciones y avanza con tu compra 🛒",
    "Aquí tienes el {label} para ti. Elige el ideal y sigue 👉",
    "Tu búsqueda de el {label} salió perfecta. Revisa y elige 📱",
    "¡Genial noticia! El {label} disponible. Mira colores y elige el tuyo ✨",
    "Encontramos el {label} para ti. ¿Listo para decidir? 👇",
    "Tenemos el {label} en Tienda Claro. Elige tu opción y continúa 🚀",
    "¿Buscas el {label}? Aquí están los mejores para decidir hoy 🔥",
]

PHONE_TITLE_SIMILAR_TEMPLATES = [
"Analizamos tu búsqueda y esto es lo más cercano a lo ideal 💡",
"Según lo que buscabas, estas opciones tienen mucho sentido 👌",
"Si miramos tu intención, estas alternativas encajan perfecto 🎯",
"Esto es lo que elegiría una IA con buen gusto 😌",
"Opciones alineadas con lo que realmente estabas buscando 🧠"
]

PHONE_DESC_EXACT_TEMPLATES = [
    "¡Lo tenemos! El {label} está disponible. Revisa opciones y sigue con tu compra ✅",
    "¡Excelente noticia! El {label} está aquí. Elige el plan que te conviene y sigue 👇",
    "Encontramos el {label} para ti. Compara precios y elige tu favorito 📱",
    "¡Sí hay! El {label} está disponible con opciones de plan. Elige y avanza 🚀",
    "Tu búsqueda dio resultado: el {label}. Mira detalles y continúa 🛒",
    "Tenemos el {label} listo para ti. Mira opciones y elige el ideal 👉",
    "¡Qué bueno! El {label} está disponible. Compara y decide hoy ✨",
    "Buenas noticias: el {label} está aquí. Elige tu opción y sigue 🔥",
    "Listo: el {label} está disponible. Revisa colores y planes 👇",
    "Aquí está el {label}. Elige tu opción favorita y continúa ✅",
]

PHONE_DESC_SIMILAR_TEMPLATES = [
    "No encontramos el {label} exacto, pero estas opciones similares te van a encantar. Elige la que más te guste 👇",
    "No está el {label} exacto, pero hay alternativas muy cercanas. Revisa y elige tu favorita ✅",
    "No vimos el {label} exacto, pero estas opciones encajan perfecto. Compara y decide 📱",
    "El {label} exacto no aparece ahora, pero aquí tienes modelos similares para elegir 👇",
    "No hay coincidencia exacta del {label}, pero estas opciones están buenísimas. Elige y continúa 🛒",
    "No encontramos el {label} exacto, pero hay alternativas top. Mira y decide 🚀",
    "No está el {label} exacto por ahora, pero estas opciones son excelentes. Elige la tuya ✨",
    "No tenemos el {label} exacto, pero estos modelos similares te van a gustar. Elige uno ✅",
    "No apareció el {label} exacto, pero aquí tienes opciones muy cercanas. Revisa y continúa 👇",
    "No hay el {label} exacto, pero estas alternativas son ideales para ti. Compara y decide 🔥",
]

PHONE_DESC_NOT_FOUND_TEMPLATES = [
    "No encontramos ese modelo exacto, pero aquí tienes alternativas increíbles para elegir 👇",
    "Por ahora no está ese modelo, pero te dejamos opciones similares para decidir ✅",
    "No apareció ese modelo exacto, pero estas alternativas pueden encantarte. Mira y elige 📱",
    "Aún no encontramos ese modelo, pero aquí tienes buenas opciones para continuar 🛒",
    "No está disponible ese modelo exacto, pero estas opciones están buenísimas. Elige la tuya 🚀",
    "No vimos ese modelo exacto, pero puedes comparar estas alternativas ahora mismo 👇",
    "Ese modelo no aparece por ahora, pero estas opciones son muy recomendadas. Elige y sigue ✨",
    "No encontramos ese modelo exacto, pero hay opciones parecidas para ti. Revisa y decide ✅",
    "Aún no tenemos ese modelo exacto, pero aquí tienes alternativas top. Compara y elige 🔥",
    "No está ese modelo exacto en este momento, pero estas opciones pueden encajar perfecto. Mira y continúa 👇",
]

def _pick_phone_title(label: str, seed: str, templates: List[str]) -> str:
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(templates)
    return templates[idx].format(label=label)

def _pick_phone_desc(label: str, seed: str, templates: List[str]) -> str:
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(templates)
    desc = templates[idx].format(label=label)
    follow_ups = [
        "¿Te gustaría que filtre por color o almacenamiento?",
        "¿Quieres que te muestre opciones con un plan específico?",
        "¿Buscas algún color en especial?",
        "¿Quieres ver alternativas más económicas o premium?",
        "¿Te gustaría comparar con otro modelo?",
    ]
    follow_idx = int(digest, 16) % len(follow_ups)
    return f"{desc} {follow_ups[follow_idx]}"

def _pick_brand_description(brand_label: str, seed: str) -> str:
    if not brand_label:
        return ""
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(BRAND_RESPONSE_TEMPLATES)
    return BRAND_RESPONSE_TEMPLATES[idx].format(brand=brand_label)

def _is_brand_only_query(nq: str) -> bool:
    if _match_any(PHONE_MODEL_PATTERNS, nq): return False
    if _nums_from_query(nq): return False
    brand = _canonical_brand_from_query(nq)
    if not brand: return False
    toks = _terms_from_query(nq)
    if not toks: return False
    allowed = {"apple","iphone","samsung","galaxy","huawei","motorola","moto",
               "xiaomi","redmi","poco","google","pixel","honor","infinix","tecno",
               "oppo","realme","vivo","lenovo","zte"}
    return all(t in allowed for t in toks)

def _has_prepago_term(nq: str) -> bool:
    return bool(re.search(r"\bprepago\b", nq, re.I))

def _explicit_modality_from_query(nq: str) -> Optional[str]:
    s = nq.lower()
    if "portabilidad" in s: return "portabilidad"
    if "renovacion" in s or "renovación" in s: return "renovación"
    if "linea nueva" in s or "línea nueva" in s: return "línea nueva"
    if "liberado" in s or "liberados" in s: return "liberados"
    return None

def _storage_from_query(nq: str) -> Optional[str]:
    m = re.search(r"\b(\d{2,4})\s*gb\b", nq, re.I)
    return m.group(1) if m else None

def _query_mentions_economy(nq: str) -> bool:
    return any(w in nq.lower() for w in ["económico","economico","barato","más barato","mas barato","precio bajo"])

def _extract_main_number(nq: str) -> Optional[str]:
    m = re.search(r"\b(\d{1,3})\b", nq)
    if m:
        return m.group(1)
    m = re.search(r"[a-z]\s*([0-9]{1,3})", nq)
    return m.group(1) if m else None

def _samsung_series_from_query(nq: str) -> Optional[str]:
    s = nq.lower()
    if re.search(r"\bgalaxy\s*s\s*\d{1,2}\b|\bs\s*\d{1,2}\b", s): return "s"
    if re.search(r"\bgalaxy\s*a\s*\d{1,2}\b|\ba\s*\d{1,2}\b", s): return "a"
    if re.search(r"\bgalaxy\s*m\s*\d{1,2}\b|\bm\s*\d{1,2}\b", s): return "m"
    if re.search(r"\bz\s*(flip|fold)\b", s): return "z"
    return None

def _pick_by_modality_priority(items: List[dict], k: int = 5, need_porta: int = 2, allowed_modalities: Optional[List[str]] = None) -> List[dict]:
    def mod_rank(m: str) -> int:
        if not m: return 99
        m = m.lower()
        if "renov" in m: return 0
        if "portabilidad" in m: return 1
        if "linea nueva" in m or "línea nueva" in m: return 2
        if "liberado" in m: return 3
        return 98

    def plan_7990_rank(r: dict) -> int:
        plan = (r.get("plan") or "").lower()
        line = (r.get("line") or "").lower()
        if line == 'postpago' and re.search(r'\bmax\b.*\bilimitado\b.*\b79[\.,]90\b', plan):
             return 0
        return 1
    
    pool = items
    pool = [item for item in pool if (item.get("line") or "").lower() in ['postpago', 'prepago'] and 'negocios' not in (item.get("line") or "").lower()]

    if allowed_modalities:
        pool = [item for item in pool if (item.get("modality") or "") in allowed_modalities or any(am.lower() in (item.get("modality","").lower()) for am in allowed_modalities)]
        if not pool:
            pool = items

    pool_sorted = sorted(pool, key=lambda r: (mod_rank(r.get("modality","")), plan_7990_rank(r), r.get("price_list") or 9e18))
    
    renovacion_candidates = [r for r in pool_sorted if 'renovación' in r.get("modality", "").lower() or 'renovacion' in r.get("modality","").lower()]
    portabilidad_candidates = [r for r in pool_sorted if 'portabilidad' in r.get("modality", "").lower()]
    
    final_items = []
    final_items.extend(renovacion_candidates[:3])
    
    portabilidad_selected = []
    selected_ids = set(item['id'] for item in final_items)
    for p in portabilidad_candidates:
        if p['id'] not in selected_ids and len(portabilidad_selected) < 2:
            portabilidad_selected.append(p)
            selected_ids.add(p['id'])
    final_items.extend(portabilidad_selected)
    
    if len(final_items) < k:
        remaining_pool = [r for r in pool_sorted if r['id'] not in selected_ids]
        final_items.extend(remaining_pool[:k - len(final_items)])
        
    return final_items[:k]

def _category_hint_from_query(nq: str) -> str:
    s = nq.lower()
    if "ipad" in s: return "ipad"
    if "tablet" in s: return "tablet"
    if "smartwatch" in s or "reloj inteligente" in s or "watch" in s: return "smartwatch"
    if "power bank" in s or "powerbank" in s or "bateria externa" in s or "batería externa" in s: return "powerbank"
    if "cargador" in s and ("inalambr" in s or "inalámbr" in s): return "charg_wireless"
    if "audifono" in s or "audifonos" in s or "audífono" in s or "audífonos" in s or "earbuds" in s or "airpods" in s or "earpods" in s or "auriculares" in s: return "headphones"
    return ""

# ------------------ BQ: Productos ------------------
def search_products_bq(nq: str, prioritize_plan_79: bool):
    terms = _terms_from_query(nq)
    #if not terms: return [], False, nq
    if not terms: return [], False, _display_query_for_phone(nq)

    want_prepago = _has_prepago_term(nq)
    explicit_mod = _explicit_modality_from_query(nq)
    storage = _storage_from_query(nq) or ""
    cheap = _query_mentions_economy(nq)
    qnum = _extract_main_number(nq)
    series = _samsung_series_from_query(nq)
    color_key, color_regex = _color_from_query(nq)

    canon = _canonical_brand_from_query(nq) or ""
    brand_aliases, brand_kw = [], ""
    if canon == "apple":
        brand_aliases, brand_kw = ["apple","iphone"], "iphone"
    elif canon == "samsung":
        brand_aliases, brand_kw = ["samsung","galaxy","samsung mobile","samsung electronics"], "galaxy"
    elif canon == "motorola":
        brand_aliases, brand_kw = ["motorola","moto"], "moto"
    elif canon == "xiaomi":
        brand_aliases, brand_kw = ["xiaomi","redmi","poco"], "xiaomi"
    elif canon == "google":
        brand_aliases, brand_kw = ["google","pixel"], "pixel"
    elif canon in ("huawei","honor","oppo","realme","vivo","infinix","tecno","lenovo","zte","tcl","nokia"):
        brand_aliases, brand_kw = [canon], canon

    brand_lock = bool(canon) 
    other_terms = OTHER_BRAND_TERMS.get(canon, [])
    force_prepago = want_prepago
    
    want_pro_max = 1 if re.search(r'\bpro\s*max\b', nq, re.I) else 0
    want_pro     = 1 if (re.search(r'\bpro\b', nq, re.I) and not want_pro_max) else 0
    want_ultra   = 1 if re.search(r'\bultra\b', nq, re.I) else 0
    want_plus    = 1 if re.search(r'(\bplus\b|\b\+\b)', nq, re.I) else 0
    want_lite    = 1 if re.search(r'\blite\b', nq, re.I) else 0
    want_fe      = 1 if re.search(r'\bfe\b', nq, re.I) else 0
    want_mini    = 1 if re.search(r'\bmini\b', nq, re.I) else 0
    want_se      = 1 if re.search(r'\bse\b', nq, re.I) else 0

    strict_sql = f"""
    DECLARE qnum STRING DEFAULT @qnum;

    WITH base AS (
      SELECT
        id, brand, title_product, image, calltoaction_url, line, modality, financing,
        title_plan,
        SAFE_CAST(NULLIF(CAST(price_list AS STRING), '') AS FLOAT64)   AS price_list,
        SAFE_CAST(NULLIF(CAST(discount AS STRING), '') AS FLOAT64)     AS discount,
        SAFE_CAST(NULLIF(CAST(price AS STRING), '') AS FLOAT64)        AS price,
        SAFE_CAST(NULLIF(CAST(price_real AS STRING), '') AS FLOAT64)   AS price_real,
        SAFE_CAST(NULLIF(CAST(price_con_descuento AS STRING), '') AS FLOAT64) AS price_con_descuento
      FROM `{BQ_TABLE_PRODUCTS}`
    ),
    brand_guard AS (
      SELECT * FROM base
      WHERE
        (@brand_lock = FALSE)
        OR (
            (
              LOWER(TRIM(IFNULL(brand,''))) IN UNNEST(@brand_aliases)
              OR (@brand_kw != '' AND STRPOS(LOWER(IFNULL(title_product,'')), @brand_kw) > 0)
            )
          AND NOT EXISTS (
              SELECT 1
              FROM UNNEST(@other_terms) AS o
              WHERE STRPOS(LOWER(IFNULL(title_product,'')), o) > 0
                OR STRPOS(LOWER(IFNULL(brand,'')), o) > 0
          )
        )
    ),
    filtered AS (
      SELECT * FROM brand_guard
      WHERE
        LOWER(IFNULL(line,'')) IN ('postpago','negocios','prepago')
        AND (
          @force_prepago = TRUE
          OR LOWER(IFNULL(modality,'')) IN ('portabilidad','renovación','renovacion')
        )
        AND (@want_prepago = FALSE OR REGEXP_CONTAINS(LOWER(IFNULL(title_plan,'')), r'prepago[[:space:]]+chevere'))
        AND (@storage = '' OR REGEXP_CONTAINS(LOWER(IFNULL(title_product,'')),
            r'(^|[^0-9])' || @storage || r'gb([^0-9]|$)'))
        AND (
          @explicit_mod IS NULL OR @explicit_mod = '' OR
          LOWER(IFNULL(modality,'')) = LOWER(@explicit_mod)
        )
        AND (@color_regex = '' OR REGEXP_CONTAINS(LOWER(IFNULL(title_product,'')), @color_regex))
    ),
    enriched_base AS (
      SELECT
        *,
        LOWER(IFNULL(title_product,'')) AS tp_lc,
        REGEXP_REPLACE(LOWER(IFNULL(title_product,'')), r'[^a-z0-9 ]', '') AS title_norm,
        REGEXP_REPLACE(
          REGEXP_REPLACE(LOWER(IFNULL(title_product,'')), r'[^a-z0-9 ]', ''),
          r'\\b(black|blue|white|green|gold|pink|silver|gray|grey|titanium|natural|negro|azul|blanco|verde|dorado|rosa|plata|gris|titanio|graphite|awesome)\\b',
          ''
        ) AS title_no_color,
        REGEXP_REPLACE(
          REGEXP_REPLACE(LOWER(IFNULL(title_product,'')), r'[^a-z0-9 ]', ''),
          r'(\\d+)\\s*gb', r'\\1gb'
        ) AS title_norm_gb,

        IF(
          @prio79 = TRUE AND @want_prepago = FALSE AND
          REGEXP_CONTAINS(LOWER(IFNULL(title_plan,'')), r'\\bmax[[:space:]]+ilimitado\\b') AND
          REGEXP_CONTAINS(LOWER(IFNULL(title_plan,'')), r'\\b79[\\., ]?90\\b') AND
          NOT REGEXP_CONTAINS(LOWER(IFNULL(line,'')), r'negocios'),
          1, 0
        ) AS plan_max_7990,

        SPLIT(
          TRIM(REGEXP_REPLACE(LOWER(IFNULL(title_product,'')), r'[^0-9]+', ' ')),
          ' '
        )[SAFE_OFFSET(0)] AS model_number_str,

        CASE
          WHEN @storage = '' THEN 0
          WHEN REGEXP_CONTAINS(LOWER(IFNULL(title_product,'')), r'(^|[^0-9])' || @storage || r'gb([^0-9]|$)') THEN 1
          ELSE 0
        END AS cap_hit
      FROM filtered
    ),
    enriched_base2 AS (
      SELECT
        enriched_base.*,
        SAFE_CAST(NULLIF(model_number_str,'') AS INT64) AS model_number
      FROM enriched_base
    ),
    enriched AS (
      SELECT
        enriched_base2.*,
        IF(REGEXP_CONTAINS(tp_lc, r'\\biphone\\s*se\\b'),1,0) AS tok_iphone_se,
        IF(REGEXP_CONTAINS(tp_lc, r'\\biphone[[:space:]]*17[[:space:]]*pro[[:space:]]*max\\b'),1,0) AS tok_ip17promax,
        IF(REGEXP_CONTAINS(tp_lc, r'\\biphone[[:space:]]*17[[:space:]]*pro\\b'),1,0) AS tok_ip17pro,
        IF(REGEXP_CONTAINS(tp_lc, r'\\biphone[[:space:]]*17[[:space:]]*plus\\b'),1,0) AS tok_ip17plus,
        IF(REGEXP_CONTAINS(tp_lc, r'\\biphone[[:space:]]*17\\b') AND NOT REGEXP_CONTAINS(tp_lc, r'\\b(pro|plus|max|ultra|mini|se|lite|fe)\\b'),1,0) AS tok_ip17,
        IF(REGEXP_CONTAINS(tp_lc, r'\\biphone[[:space:]]*16[[:space:]]*pro[[:space:]]*max\\b'),1,0) AS tok_ip16promax,
        IF(REGEXP_CONTAINS(tp_lc, r'\\biphone[[:space:]]*16[[:space:]]*pro\\b'),1,0) AS tok_ip16pro,
        IF(REGEXP_CONTAINS(tp_lc, r'\\biphone[[:space:]]*16\\b') AND NOT REGEXP_CONTAINS(tp_lc, r'\\b(pro|plus|max|ultra|mini|se|lite|fe)\\b'),1,0) AS tok_ip16,
        IF(REGEXP_CONTAINS(tp_lc, r'\\biphone[[:space:]]*15\\b') AND NOT REGEXP_CONTAINS(tp_lc, r'\\b(pro|plus|max|ultra|mini|se|lite|fe)\\b'),1,0) AS tok_ip15,
        IF(REGEXP_CONTAINS(tp_lc, r'\\biphone[[:space:]]*14\\b') AND NOT REGEXP_CONTAINS(tp_lc, r'\\b(pro|plus|max|ultra|mini|se|lite|fe)\\b'),1,0) AS tok_ip14,

        IF(REGEXP_CONTAINS(tp_lc, r'\\bmini\\b'),1,0) AS has_mini,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bpro[[:space:]]*max\\b'),1,0) AS has_pro_max,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bpro\\b') AND NOT REGEXP_CONTAINS(tp_lc, r'\\bmax\\b'),1,0) AS has_pro_only,
        IF(REGEXP_CONTAINS(tp_lc, r'\\b\\+\\b|\\bplus\\b'),1,0) AS has_plus,
        IF(REGEXP_CONTAINS(tp_lc, r'\\blite\\b'),1,0) AS has_lite,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bfe\\b'),1,0) AS has_fe,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bse\\b'),1,0) AS has_se,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bultra\\b'),1,0) AS has_ultra,

        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*s[[:space:]]*25[[:space:]]*ultra\\b'),1,0) AS tok_s25ultra,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*s[[:space:]]*25[[:space:]]*\\+\\b|\\bgalaxy\\s*s\\s*25\\s*plus\\b'),1,0) AS tok_s25plus,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*s[[:space:]]*25[[:space:]]*fe\\b'),1,0) AS tok_s25fe,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*s[[:space:]]*25\\b') AND NOT REGEXP_CONTAINS(tp_lc, r'\\b(ultra|plus|fe)\\b'),1,0) AS tok_s25,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*s[[:space:]]*24[[:space:]]*ultra\\b'),1,0) AS tok_s24ultra,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*s[[:space:]]*24[[:space:]]*\\+\\b|\\bgalaxy\\s*s\\s*24\\s*plus\\b'),1,0) AS tok_s24plus,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*s[[:space:]]*24\\b') AND NOT REGEXP_CONTAINS(tp_lc, r'\\b(ultra|plus|fe)\\b'),1,0) AS tok_s24,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*a[[:space:]]*56\\b'),1,0) AS tok_a56,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*a[[:space:]]*55\\b'),1,0) AS tok_a55,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*a[[:space:]]*36\\b'),1,0) AS tok_a36,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*a[[:space:]]*35\\b'),1,0) AS tok_a35,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*a[[:space:]]*17\\b'),1,0) AS tok_a17,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*a[[:space:]]*16\\b'),1,0) AS tok_a16,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*a[[:space:]]*15\\b'),1,0) AS tok_a15,

        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*(s|a|m)[[:space:]]*[0-9]{1,2}\\b'), 1, 0) AS famnum_samsung,
        IF(REGEXP_CONTAINS(tp_lc, r'\\biphone[[:space:]]*[0-9]{1,2}\\b'), 1, 0) AS famnum_iphone
      FROM enriched_base2
    ),
    scored AS (
      SELECT
        e.*,
        CASE
          WHEN @want_pro=1 THEN CASE
            WHEN has_pro_only=1 THEN 0
            WHEN has_pro_max=1  THEN -120 
            ELSE -60
          END
          WHEN @want_plus=1 THEN CASE
            WHEN has_plus=1 THEN 0
            WHEN has_ultra=1 THEN -120 
            ELSE -60
          END
          WHEN @want_ultra=1    THEN CASE WHEN has_ultra=1    THEN 0 ELSE -120 END
          WHEN @want_pro_max=1 THEN CASE WHEN has_pro_max=1  THEN 0 ELSE -120 END
          WHEN @want_lite=1     THEN CASE WHEN has_lite=1     THEN 0 ELSE -90  END
          WHEN @want_fe=1       THEN CASE WHEN has_fe=1       THEN 0 ELSE -90  END
          WHEN @want_mini=1     THEN CASE WHEN has_mini=1     THEN 0 ELSE -90  END
          WHEN @want_se=1       THEN CASE WHEN has_se=1       THEN 0 ELSE -90  END
          ELSE CASE
            WHEN (has_pro_max=1 OR has_pro_only=1 OR has_ultra=1 OR has_plus=1 OR has_lite=1 OR has_fe=1 OR has_mini=1 OR has_se=1)
            THEN -60 ELSE 0
          END
        END AS variant_penalty,
        CASE
          WHEN @brand_lock = TRUE AND @brand_kw = 'iphone' AND @qnum IS NOT NULL AND @qnum != '' THEN
            CASE
              WHEN e.model_number IS NULL THEN 0 
              WHEN CAST(@qnum AS INT64) = e.model_number THEN 0 
              ELSE -80
            END
          ELSE 0
        END AS apple_num_mismatch_penalty,
        (
          50 * (
            e.tok_ip17promax + e.tok_ip17pro + e.tok_ip17plus + e.tok_ip17 +
            e.tok_ip16promax + e.tok_ip16pro + e.tok_ip16 + e.tok_ip15 + e.tok_ip14 +
            e.tok_s25ultra + e.tok_s25plus + e.tok_s25fe + e.tok_s25 +
            e.tok_s24ultra + e.tok_s24plus + e.tok_s24 +
            e.tok_a56 + e.tok_a55 + e.tok_a36 + e.tok_a35 + e.tok_a17 + e.tok_a16 + e.tok_a15
          )
          + 10 * (e.famnum_samsung + e.famnum_iphone)
          + 3  * IF(@prio79, e.plan_max_7990, 0)
          + 8  * e.cap_hit
          + 1  * ARRAY_LENGTH(ARRAY(SELECT t FROM UNNEST(@terms) AS t WHERE STRPOS(e.tp_lc, t) > 0))
          + (CASE WHEN @want_pro=1 THEN (CASE WHEN e.has_pro_only=1 THEN 0 ELSE -120 END) WHEN @want_pro_max=1 THEN (CASE WHEN e.has_pro_max=1 THEN 0 ELSE -120 END) ELSE 0 END)
          + (CASE WHEN @brand_lock = TRUE AND @brand_kw = 'iphone' AND @qnum IS NOT NULL AND @qnum != '' THEN (CASE WHEN e.model_number IS NULL THEN 0 WHEN CAST(@qnum AS INT64) = e.model_number THEN 0 ELSE -80 END) ELSE 0 END)
        ) AS match_score,
        IF(
          @qnum IS NULL OR @qnum = '',
          0,
          CASE WHEN REGEXP_CONTAINS(e.title_norm, r'(^|[^0-9])' || @qnum || r'([^0-9]|$)') THEN 2
               WHEN STRPOS(e.title_norm, @qnum) > 0 THEN 1
               ELSE 0
          END
        ) AS number_soft_score
      FROM enriched e
    )
    SELECT *
    FROM scored
    WHERE
      (@qnum IS NULL OR @qnum = '' OR number_soft_score >= 1)
      AND (@series = '' OR
            (@series = 'z' AND REGEXP_CONTAINS(title_norm, r'\\bz[[:space:]]?(flip|fold)\\b')) OR
            (@series = 's' AND REGEXP_CONTAINS(title_norm, r'\\bgalaxy\\b') AND REGEXP_CONTAINS(title_norm, 's' || @qnum)) OR
            (@series = 'a' AND REGEXP_CONTAINS(title_norm, r'\\bgalaxy\\b') AND REGEXP_CONTAINS(title_norm, 'a' || @qnum)) OR
            (@series = 'm' AND REGEXP_CONTAINS(title_norm, r'\\bgalaxy\\b') AND REGEXP_CONTAINS(title_norm, 'm' || @qnum))
      )
    ORDER BY
      (tok_ip17promax + tok_ip17pro + tok_ip17plus + tok_ip17 +
       tok_ip16promax + tok_ip16pro + tok_ip16 + tok_ip15 + tok_ip14 +
       tok_s25ultra + tok_s25plus + tok_s25fe + tok_s25 +
       tok_s24ultra + tok_s24plus + tok_s24 +
       tok_a56 + tok_a55 + tok_a36 + tok_a35 + tok_a17 + tok_a16 + tok_a15) DESC,
      variant_penalty DESC,
      plan_max_7990  DESC,
      match_score DESC,
      COALESCE(price_list, 9e18) ASC
    LIMIT 80
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("qnum", "STRING", qnum or ""),
            bigquery.ArrayQueryParameter("terms", "STRING", terms),
            bigquery.ScalarQueryParameter("series", "STRING", series or ""),
            bigquery.ScalarQueryParameter("prio79", "BOOL", prioritize_plan_79 and not want_prepago),
            bigquery.ScalarQueryParameter("want_prepago", "BOOL", want_prepago),
            bigquery.ScalarQueryParameter("explicit_mod", "STRING", explicit_mod or ""),
            bigquery.ScalarQueryParameter("force_prepago", "BOOL", force_prepago),
            bigquery.ScalarQueryParameter("brand_lock", "BOOL", bool(brand_lock)),
            bigquery.ArrayQueryParameter("brand_aliases", "STRING", brand_aliases or []),
            bigquery.ScalarQueryParameter("brand_kw", "STRING", brand_kw),
            bigquery.ArrayQueryParameter("other_terms", "STRING", other_terms),
            bigquery.ScalarQueryParameter("storage", "STRING", storage),
            bigquery.ScalarQueryParameter("cheap", "BOOL", cheap),
            bigquery.ScalarQueryParameter("color_regex", "STRING", color_regex or ""),
            bigquery.ScalarQueryParameter("want_pro_max", "INT64", want_pro_max),
            bigquery.ScalarQueryParameter("want_pro", "INT64", want_pro),
            bigquery.ScalarQueryParameter("want_ultra", "INT64", want_ultra),
            bigquery.ScalarQueryParameter("want_plus", "INT64", want_plus),
            bigquery.ScalarQueryParameter("want_lite", "INT64", want_lite),
            bigquery.ScalarQueryParameter("want_fe", "INT64", want_fe),
            bigquery.ScalarQueryParameter("want_mini", "INT64", want_mini),
            bigquery.ScalarQueryParameter("want_se", "INT64", want_se),
        ]
    )
    rows = list(bq_client.query(strict_sql, job_config=job_config).result())

    if not rows:
        relaxed_sql = f"""
        DECLARE qnum STRING DEFAULT @qnum;
        WITH base AS (
          SELECT
            id, brand, title_product, image, calltoaction_url, line, modality, financing,
            title_plan,
            SAFE_CAST(NULLIF(CAST(price_list AS STRING), '') AS FLOAT64)   AS price_list,
            SAFE_CAST(NULLIF(CAST(discount AS STRING), '') AS FLOAT64)     AS discount
          FROM `{BQ_TABLE_PRODUCTS}`
        ),
        brand_guard AS (
          SELECT * FROM base
          WHERE
            (@brand_lock = FALSE)
            OR (
                (
                  LOWER(TRIM(IFNULL(brand,''))) IN UNNEST(@brand_aliases)
                  OR (@brand_kw != '' AND STRPOS(LOWER(IFNULL(title_product,'')), @brand_kw) > 0)
                )
                AND NOT EXISTS (
                  SELECT 1 FROM UNNEST(@other_terms) AS o
                  WHERE STRPOS(LOWER(IFNULL(title_product,'')), o) > 0
                     OR STRPOS(LOWER(IFNULL(brand,'')), o) > 0
                )
            )
        ),
        filtered AS (
          SELECT * FROM brand_guard
          WHERE
            LOWER(IFNULL(line,'')) IN ('postpago','negocios','prepago')
            AND (
              @force_prepago = TRUE
              OR LOWER(IFNULL(modality,'')) IN ('portabilidad','renovación','renovacion')
            )
            AND (@color_regex = '' OR REGEXP_CONTAINS(LOWER(IFNULL(title_product,'')), @color_regex))
        ),
        enriched AS (
          SELECT
            *,
            REGEXP_REPLACE(LOWER(IFNULL(title_product,'')), r'[^a-z0-9 ]', '') AS title_norm,
            IF(
              @prio79 = TRUE AND @want_prepago = FALSE AND
              REGEXP_CONTAINS(LOWER(IFNULL(title_plan,'')), r'\\bmax[[:space:]]+ilimitado\\b') AND
              REGEXP_CONTAINS(LOWER(IFNULL(title_plan,'')), r'\\b79[\\., ]?90\\b') AND
              NOT REGEXP_CONTAINS(LOWER(IFNULL(line,'')), r'negocios'),
              1, 0
            ) AS plan_max_7990
          FROM filtered
        ),
        scored AS (
          SELECT
            *,
            CASE
              WHEN @qnum IS NULL OR @qnum = '' THEN 0
              WHEN REGEXP_CONTAINS(title_norm, r'(^|[^0-9])' || @qnum || r'([^0-9]|$)') THEN 2
              WHEN STRPOS(title_norm, @qnum) > 0 THEN 1
              ELSE 0
            END AS number_soft_score
          FROM enriched
        )
        SELECT *
        FROM scored
        WHERE (@qnum IS NULL OR @qnum = '' OR number_soft_score >= 1)
        ORDER BY plan_max_7990 DESC, COALESCE(price_list, 9e18) ASC
        LIMIT 80
        """
        rows = list(bq_client.query(relaxed_sql, job_config=job_config).result())

    had_exact = bool(rows) and bool(qnum)

    if (not rows or not had_exact) and (canon and qnum and qnum.isdigit()):
        q = int(qnum)
        near_nums = [str(max(1, q-2)), str(max(1, q-1)), str(q+1), str(q+2)]
        near_nums = [n for n in near_nums if 10 <= int(n) <= 30] 
        near_sql = f"""
        WITH base AS (
          SELECT id, brand, title_product, image, calltoaction_url, line, modality, financing,
                 title_plan,
                 SAFE_CAST(NULLIF(CAST(price_list AS STRING), '') AS FLOAT64) AS price_list,
                 SAFE_CAST(NULLIF(CAST(discount AS STRING), '') AS FLOAT64)   AS discount
          FROM `{BQ_TABLE_PRODUCTS}`
          WHERE LOWER(TRIM(IFNULL(brand,''))) IN UNNEST(@brand_aliases)
        ),
        e AS (
          SELECT *,
                 REGEXP_REPLACE(LOWER(IFNULL(title_product,'')), r'[^a-z0-9 ]', '') AS title_norm
          FROM base
        )
        SELECT *
        FROM e
        WHERE
          REGEXP_CONTAINS(title_norm, r'(^|[^0-9])(' || ARRAY_TO_STRING(@near_nums, '|') || r')([^0-9]|$)')
          AND (@color_regex = '' OR REGEXP_CONTAINS(title_norm, @color_regex))
        ORDER BY COALESCE(price_list,9e18) ASC
        LIMIT 20
        """
        jc2 = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("brand_aliases", "STRING", brand_aliases or []),
                bigquery.ArrayQueryParameter("near_nums", "STRING", near_nums),
                bigquery.ScalarQueryParameter("color_regex", "STRING", color_regex or ""),
            ]
        )
        near_rows = list(bq_client.query(near_sql, job_config=jc2).result())
        if near_rows:
            rows = near_rows
            had_exact = False 

    cand = []
    for r in rows:
        title_full = r["title_product"] or ""
        line_val = (r["line"] or "")
        modality_val = (r["modality"] or "")
        plan_val = (r["title_plan"] or "")
        cand.append({
            "id": r["id"], "marca": r["brand"], 
            #"producto": r["title_product"][:38],
            "producto": title_full[:38],
            "imagen": r["image"],
            "url": r["calltoaction_url"], "tipo": "Celular",
            "plan": r["title_plan"], "price_list": r["price_list"], "discount": r["discount"],
            "porcentaje_descuento": None, "line": r["line"], "modality": r["modality"], "financing": r["financing"],
            "_color": _extract_color_from_title(title_full),
            "_storage": _extract_storage_from_title(title_full),
            "_model": _extract_model_number_from_title(title_full, canon),
            "_variant": _variant_rank(title_full),
            "_is_7990": bool(
                line_val.lower() == "postpago"
                and re.search(r'\bmax\b.*\bilimitado\b.*\b79[\.,]90\b', plan_val.lower() or "")
                and re.search(r"(portabilidad|renovacion|renovación)", modality_val.lower())
            ),
        })

    allowed = None if explicit_mod else ["Portabilidad","Renovación"] if not want_prepago else None

    top = _pick_by_modality_priority(cand, k=24, need_porta=2, allowed_modalities=allowed)
    require_strict_plan = not explicit_mod and not want_prepago
    if require_strict_plan:
        strict_top = [item for item in top if item.get("_is_7990")]
        if strict_top:
            top = strict_top

    model_specific = bool(qnum) or _match_any(PHONE_MODEL_PATTERNS, nq)
    storage_requested = bool(storage)
    target_model = int(qnum) if (qnum and qnum.isdigit()) else None
    candidates = top

    if model_specific and target_model:
        same_model = [c for c in candidates if c.get("_model") == target_model]
        if storage_requested:
            same_model = sorted(same_model, key=lambda x: x.get("_storage", 0), reverse=True)
        else:
            same_model = sorted(same_model, key=lambda x: (x.get("_variant", 0), x.get("_storage", 0)), reverse=True)

        def _pick_colors(pool, limit):
            picked, seen = [], set()
            for item in pool:
                color = item.get("_color") or ""
                if color and color in seen:
                    continue
                picked.append(item)
                if color:
                    seen.add(color)
                if len(picked) >= limit:
                    break
            return picked

        def _pick_models_colors(pool, limit):
            picked, seen_models, seen_pairs = [], set(), set()
            for item in pool:
                model = item.get("_model") or 0
                color = item.get("_color") or ""
                if model and model in seen_models:
                    continue
                if color and (model, color) in seen_pairs:
                    continue
                picked.append(item)
                if model:
                    seen_models.add(model)
                if color:
                    seen_pairs.add((model, color))
                if len(picked) >= limit:
                    break
            return picked

        selected = _pick_colors(same_model, 5)
        if len(selected) < 5:
            variants = [c for c in same_model if c not in selected]
            selected.extend(variants[:5 - len(selected)])

        if len(selected) < 5:
            nearby = [c for c in candidates if c.get("_model") and c.get("_model") != target_model]
            nearby = sorted(nearby, key=lambda x: x.get("_model", 0), reverse=True)
            selected.extend(_pick_models_colors(nearby, 5 - len(selected)))

    else:
        candidates = sorted(candidates, key=lambda x: (x.get("_model", 0), x.get("_variant", 0)), reverse=True)
        selected, seen_pairs = [], set()
        for item in candidates:
            color = item.get("_color") or ""
            model = item.get("_model") or 0
            if color and (model, color) in seen_pairs:
                continue
            selected.append(item)
            if color:
                seen_pairs.add((model, color))
            if len(selected) >= 5:
                break

    if len(selected) < 5:
        for item in candidates:
            if item in selected:
                continue
            selected.append(item)
            if len(selected) >= 5:
                break

    top = [{k: v for k, v in item.items() if not k.startswith("_")} for item in selected[:5]]
    return top, had_exact, _display_query_for_phone(nq)

def search_brand_only_bq(nq_brand: str) -> List[dict]:
    brand_focus = _canonical_brand_from_query(nq_brand)
    if not brand_focus:
        return []

    if brand_focus == "apple":
        brand_aliases, brand_kw = ["apple","iphone"], "iphone"
    elif brand_focus == "samsung":
        brand_aliases, brand_kw = ["samsung","galaxy","samsung mobile","samsung electronics"], "galaxy"
    elif brand_focus == "motorola":
        brand_aliases, brand_kw = ["motorola","moto"], "moto"
    elif brand_focus == "xiaomi":
        brand_aliases, brand_kw = ["xiaomi","redmi","poco"], "xiaomi"
    elif brand_focus == "google":
        brand_aliases, brand_kw = ["google","pixel"], "pixel"
    else:
        brand_aliases, brand_kw = [brand_focus], brand_focus

    sql = f"""
    SELECT
      id, brand, title_product, image, calltoaction_url, line, modality, financing,
      title_plan,
      SAFE_CAST(NULLIF(CAST(price_list AS STRING), '') AS FLOAT64) AS price_list,
      SAFE_CAST(NULLIF(CAST(discount AS STRING), '') AS FLOAT64)   AS discount
    FROM `{BQ_TABLE_PRODUCTS}`
    WHERE
      (
        LOWER(TRIM(IFNULL(brand,''))) IN UNNEST(@brand_aliases)
        OR (@brand_kw != '' AND STRPOS(LOWER(IFNULL(title_product,'')), @brand_kw) > 0)
      )
      AND LOWER(IFNULL(line,'')) IN ('postpago','prepago')
      AND REGEXP_CONTAINS(LOWER(IFNULL(modality,'')), r'(portabilidad|renovacion|renovación)')
      AND REGEXP_CONTAINS(LOWER(IFNULL(title_plan,'')), r'\\bmax[[:space:]]+ilimitado\\b')
      AND REGEXP_CONTAINS(LOWER(IFNULL(title_plan,'')), r'\\b79[\\., ]?90\\b')
    LIMIT 120
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("brand_aliases", "STRING", brand_aliases),
            bigquery.ScalarQueryParameter("brand_kw", "STRING", brand_kw),
        ]
    )
    rows = list(bq_client.query(sql, job_config=job_config).result())
    if not rows:
        relaxed_sql = f"""
        SELECT
          id, brand, title_product, image, calltoaction_url, line, modality, financing,
          title_plan,
          SAFE_CAST(NULLIF(CAST(price_list AS STRING), '') AS FLOAT64) AS price_list,
          SAFE_CAST(NULLIF(CAST(discount AS STRING), '') AS FLOAT64)   AS discount
        FROM `{BQ_TABLE_PRODUCTS}`
        WHERE
          (
            LOWER(TRIM(IFNULL(brand,''))) IN UNNEST(@brand_aliases)
            OR (@brand_kw != '' AND STRPOS(LOWER(IFNULL(title_product,'')), @brand_kw) > 0)
          )
          AND LOWER(IFNULL(line,'')) IN ('postpago','prepago')
          AND REGEXP_CONTAINS(LOWER(IFNULL(modality,'')), r'(portabilidad|renovacion|renovación)')
        LIMIT 120
        """
        rows = list(bq_client.query(relaxed_sql, job_config=job_config).result())
    if not rows:
        return []

    def _normalize_model_key(title: str) -> str:
        cleaned = strip_accents((title or "").lower())
        cleaned = re.sub(r"\b\d{2,4}\s*gb\b", "", cleaned)
        cleaned = re.sub(r"\b(5g|4g)\b", "", cleaned)
        cleaned = re.sub(
            r"\b(black|blue|white|green|gold|pink|silver|gray|grey|titanium|natural|negro|azul|blanco|verde|dorado|rosa|plata|gris|titanio|graphite|awesome)\b",
            "",
            cleaned,
        )
        cleaned = re.sub(r"[^a-z0-9 ]", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    candidates = []
    for r in rows:
        title_full = r["title_product"] or ""
        candidates.append({
            "id": r["id"], "marca": r["brand"],
            "producto": title_full[:38],
            "imagen": r["image"],
            "url": r["calltoaction_url"], "tipo": "Celular",
            "plan": r["title_plan"], "price_list": r["price_list"], "discount": r["discount"],
            "line": r["line"], "modality": r["modality"], "financing": r["financing"],
            "_model": _extract_model_number_from_title(title_full, brand_focus),
            "_variant": _variant_rank(title_full),
            "_storage": _extract_storage_from_title(title_full),
            "_color": _extract_color_from_title(title_full),
            "_model_key": _normalize_model_key(title_full),
        })

    candidates.sort(key=lambda x: (x["_model"], x.get("_variant", 0), x["_storage"]), reverse=True)

    dedupe_by_variant = brand_focus != "samsung"
    picked, seen_keys, seen_pairs, seen_model_keys = [], set(), set(), set()
    for item in candidates:
        variant_key = item.get("_variant", 0) if dedupe_by_variant else 0
        model_key = item.get("_model_key") or ""
        key = (item["_model"], variant_key)
        if item["_model"] and key in seen_keys:
            continue
        if brand_focus == "samsung" and model_key and model_key in seen_model_keys:
            continue
        if item["_color"] and (item["_model"], variant_key, item["_color"]) in seen_pairs:
            continue
        picked.append(item)
        if item["_model"]:
            seen_keys.add(key)
        if item["_color"]:
            seen_pairs.add((item["_model"], variant_key, item["_color"]))
        if model_key:
            seen_model_keys.add(model_key)
        if len(picked) >= 5:
            break

    if len(picked) < 5:
        for item in candidates:
            if item in picked:
                continue
            picked.append(item)
            if len(picked) >= 5:
                break

    return [{k: v for k, v in item.items() if not k.startswith("_")} for item in picked[:5]]
# ------------------ BQ: Marca-solo (TOP5 último mes) ------------------
# [SECCION REINTEGRADA: AUXILIARES PARA TOP 5 MARCAS]
def _last_closed_month_start_date_lima() -> datetime.date:
    now_lima = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-5)))
    first_this_month = now_lima.replace(day=1).date()
    last_month_end = first_this_month - datetime.timedelta(days=1)
    return last_month_end.replace(day=1)

def _month_spanish_name(m: int) -> str:
    return ["enero","febrero","marzo","abril","mayo","junio",
            "julio","agosto","septiembre","octubre","noviembre","diciembre"][m-1]

def _fetch_top5_names_by_brand_last_month(brand_focus: str) -> Tuple[List[str], str]:
    table = BQ_TABLE_TOP5
    brand = (brand_focus or "").lower()
    lm = _last_closed_month_start_date_lima()
    lm_str = lm.isoformat()

    sql_last_closed = f"""
    WITH base AS (
      SELECT producto
      FROM `{table}`
      WHERE mes = DATE(@lm)
        AND LOWER(TRIM(marca)) = @brand
      ORDER BY ingresos_totales DESC
      LIMIT 5
    )
    SELECT producto FROM base
    """
    job_cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("lm", "DATE", lm_str),
            bigquery.ScalarQueryParameter("brand", "STRING", brand),
        ]
    )
    rows = list(bq_client.query(sql_last_closed, job_config=job_cfg).result())
    names = [r["producto"] for r in rows]

    if names:
        mes_label = f"{_month_spanish_name(lm.month)} {lm.year}"
        return names, mes_label

    sql_latest_for_brand = f"""
    WITH avail AS (
      SELECT mes
      FROM `{table}`
      WHERE LOWER(TRIM(marca)) = @brand
      GROUP BY mes
      ORDER BY mes DESC
      LIMIT 1
    ),
    top5 AS (
      SELECT producto
      FROM `{table}`
      WHERE mes = (SELECT mes FROM avail)
        AND LOWER(TRIM(marca)) = @brand
      ORDER BY ingresos_totales DESC
      LIMIT 5
    )
    SELECT (SELECT mes FROM avail) AS mes, producto
    FROM top5
    """
    job_cfg2 = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("brand", "STRING", brand)]
    )
    rows2 = list(bq_client.query(sql_latest_for_brand, job_config=job_cfg2).result())
    if not rows2:
        return [], ""

    mes = rows2[0]["mes"]
    mes_label = f"{_month_spanish_name(mes.month)} {mes.year}"
    names2 = [r["producto"] for r in rows2]
    return names2, mes_label

# [SECCION REINTEGRADA: FUNCION PRINCIPAL DE BUSQUEDA TOP 5]
def search_brand_top5_products(nq_brand: str) -> Tuple[List[dict], str]:
    brand_focus = _canonical_brand_from_query(nq_brand)
    if not brand_focus:
        return [], ""

    def _brand_aliases_for_top(canon: str):
        canon = (canon or "").lower().strip()
        if canon == "apple":     return ["apple","iphone"]
        if canon == "samsung":   return ["samsung","galaxy"]
        if canon == "motorola":  return ["motorola","moto"]
        if canon == "xiaomi":    return ["xiaomi","redmi","poco"]
        if canon == "google":    return ["google","pixel"]
        if canon == "lenovo":    return ["lenovo"]
        if canon in ("huawei","honor","oppo","realme","vivo","infinix","tecno","zte"): return [canon]
        return [canon] if canon else []

    brand_aliases = _brand_aliases_for_top(brand_focus)
    top_names, mes_label = _fetch_top5_names_by_brand_last_month(brand_focus)
    if not top_names:
        return [], ""

    sql = f"""
    WITH names AS (
      SELECT idx, n AS raw_name
      FROM UNNEST(@names) AS n WITH OFFSET AS idx
    ),
    norm_names AS (
      SELECT
        idx,
        raw_name,
        REGEXP_REPLACE(
          REGEXP_REPLACE(LOWER(raw_name), r'[^a-z0-9 ]', ''),
          r'\\b(black|blue|white|green|gold|pink|silver|gray|grey|titanium|natural|negro|azul|blanco|verde|dorado|rosa|plata|gris|titanio|graphite|awesome)\\b',
          ''
        ) AS name_no_color,
        REGEXP_REPLACE(
          REGEXP_REPLACE(LOWER(raw_name), r'[^a-z0-9 ]', ''),
          r'(\\d+)\\s*gb', r'\\1gb'
        ) AS name_norm_gb
      FROM names
    ),
    prod AS (
      SELECT
        p.*,
        REGEXP_REPLACE(LOWER(IFNULL(p.title_product,'')), r'[^a-z0-9 ]', '') AS title_norm
      FROM `{BQ_TABLE_PRODUCTS}` p
      WHERE
           LOWER(TRIM(IFNULL(p.brand,''))) IN UNNEST(@brand_aliases)
        OR STRPOS(LOWER(IFNULL(p.title_product,'')), @brand_kw) > 0
    ),
    prod_norm AS (
      SELECT
        *,
        REGEXP_REPLACE(
          title_norm,
          r'\\b(black|blue|white|green|gold|pink|silver|gray|grey|titanium|natural|negro|azul|blanco|verde|dorado|rosa|plata|gris|titanio|graphite|awesome)\\b',
          ''
        ) AS title_no_color,
        REGEXP_REPLACE(title_norm, r'(\\d+)\\s*gb', r'\\1gb') AS title_norm_gb
      FROM prod
    ),
    joined AS (
      SELECT
        n.idx, n.raw_name,
        p.id, p.brand, p.title_product, p.image, p.calltoaction_url,
        p.line, p.modality, p.financing, p.title_plan, p.price_list, p.discount
      FROM norm_names n
      JOIN prod_norm p
      ON  STRPOS(p.title_norm_gb, n.name_norm_gb) > 0
       OR STRPOS(p.title_no_color, n.name_no_color) > 0
    ),
    ranked AS (
      SELECT
        *,
        IF(
          REGEXP_CONTAINS(LOWER(IFNULL(title_plan,'')), r'\\bmax[[:space:]]+ilimitado\\b') AND
          REGEXP_CONTAINS(LOWER(IFNULL(title_plan,'')), r'\\b79[\\., ]?90\\b') AND
          NOT REGEXP_CONTAINS(LOWER(IFNULL(line,'')), r'negocios'),
          1, 0
        ) AS plan_max_7990
      FROM joined
    ),
    pick_one_per_name AS (
      SELECT *
      FROM (
        SELECT
          r.*,
          ROW_NUMBER() OVER (PARTITION BY raw_name ORDER BY plan_max_7990 DESC, COALESCE(price_list, 9e18) ASC) AS rn
        FROM ranked r
      )
      WHERE rn = 1
    )
    SELECT *
    FROM pick_one_per_name
    ORDER BY idx
    LIMIT 12
    """

    brand_kw = "iphone" if brand_focus=="apple" else ("galaxy" if brand_focus=="samsung" else brand_focus)
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("names", "STRING", top_names),
            bigquery.ArrayQueryParameter("brand_aliases", "STRING", brand_aliases),
            bigquery.ScalarQueryParameter("brand_kw", "STRING", brand_kw),
        ]
    )
    rows = list(bq_client.query(sql, job_config=job_config).result())

    items = [{
        "id": r["id"], "marca": r["brand"], 
        "producto": r["title_product"][:38],
        "imagen": r["image"],
        "url": r["calltoaction_url"], "tipo": "Celular", "plan": r["title_plan"],
        "price_list": r["price_list"], "discount": r["discount"],
        "line": r["line"], "modality": r["modality"], "financing": r["financing"],
    } for r in rows]

    # Relleno con 79.90 + Renovación/Portabilidad + Postpago (diversidad por modelo)
    if len(items) < 5 and brand_focus:
        missing_count = 5 - len(items)
        existing_ids = [item["id"] for item in items]
        
        fallback_sql = f"""
        WITH all_products AS (
            SELECT
                id, brand, title_product, image, calltoaction_url, line, modality, financing,
                title_plan,
                SAFE_CAST(NULLIF(CAST(price_list AS STRING), '') AS FLOAT64) AS price_list,
                SAFE_CAST(NULLIF(CAST(discount AS STRING), '') AS FLOAT64)   AS discount,
                TRIM(REGEXP_REPLACE(
                    LOWER(title_product),
                    r'(\\s(256|512|1024)gb|\\s(pro|max|ultra|plus|mini|lite|fe|se)\\b|\\s(\\d{2,4})gb\\b)',
                    ''
                )) AS model_group
            FROM `{BQ_TABLE_PRODUCTS}`
            WHERE LOWER(TRIM(IFNULL(brand,''))) IN UNNEST(@brand_aliases)
              AND LOWER(IFNULL(line,'')) = 'postpago'
              AND NOT REGEXP_CONTAINS(LOWER(IFNULL(line,'')), r'negocios')
              AND id NOT IN UNNEST(@existing_ids)
              AND REGEXP_CONTAINS(LOWER(IFNULL(modality,'')), r'(renovacion|portabilidad)')
              AND REGEXP_CONTAINS(LOWER(IFNULL(title_plan,'')), r'\\bmax[[:space:]]+ilimitado\\b')
              AND REGEXP_CONTAINS(LOWER(IFNULL(title_plan,'')), r'\\b79[\\., ]?90\\b')
        ),
        ranked_models AS (
            SELECT
                *,
                ROW_NUMBER() OVER (PARTITION BY model_group ORDER BY COALESCE(price_list, 9e18) ASC) AS rn
            FROM all_products
        )
        SELECT 
            id, brand, title_product, image, calltoaction_url, line, modality, financing,
            title_plan, price_list, discount 
        FROM ranked_models
        WHERE rn = 1
        ORDER BY COALESCE(price_list, 9e18) ASC
        LIMIT @missing_count
        """
        
        fallback_job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("brand_aliases", "STRING", _brand_aliases_for_top(brand_focus)),
                bigquery.ScalarQueryParameter("missing_count", "INT64", missing_count),
                bigquery.ArrayQueryParameter("existing_ids", "STRING", existing_ids),
            ]
        )
        fallback_rows = list(bq_client.query(fallback_sql, job_config=fallback_job_config).result())
        
        fallback_items = [{
            "id": r["id"], "marca": r["brand"], 
            "producto": r["title_product"][:38],
            "imagen": r["image"],
            "url": r["calltoaction_url"], "tipo": "Celular", "plan": r["title_plan"],
            "price_list": r["price_list"], "discount": r["discount"],
            "line": r["line"], "modality": r["modality"], "financing": r["financing"],
        } for r in fallback_rows]
        
        items.extend(fallback_items)

    items = _pick_by_modality_priority(items, k=5, need_porta=2)
    return items, mes_label

# ------------------ Accesorios / tienda_productos ------------------
# [OPTIMIZACIÓN] Limit aumentado a 50 en SQL
def _run_accessories_query(nq_terms: List[str], brand_focus: Optional[str], require_brand: bool, strict_brand: bool = False, cat_hint: str = "") -> List[dict]:
    if not nq_terms:
        return []
    sql = f"""
    WITH base AS (
      SELECT
        id, brand, title_product, image, calltoaction_url,
        line, modality, financing,
        title_plan, price_list, discount,
        ARRAY<STRING>[] AS keywords
      FROM `{BQ_TABLE_PRODUCTS}`
    ),
    enriched AS (
      SELECT
        id, brand, title_product, image, calltoaction_url,
        line, modality, financing,
        title_plan, price_list, discount,
        CASE
          WHEN price_list IS NOT NULL AND price_list > 0 AND discount IS NOT NULL
            THEN GREATEST(0, LEAST(100, ROUND(((price_list - discount) / price_list) * 100)))
          ELSE NULL
        END AS porcentaje_descuento,
        IF(
          REGEXP_CONTAINS(LOWER(IFNULL(title_product,'')), r'(iphone|galaxy|xiaomi|redmi|poco|huawei|motorola|pixel|honor|infinix|tecno|moto\\s)') OR
          REGEXP_CONTAINS(LOWER(IFNULL(line,'')), r'(equipo|celular|postpago|prepago)'),
          1, 0
        ) AS es_celular,
        ARRAY_LENGTH(ARRAY(
          SELECT a FROM UNNEST(@acc) AS a
          WHERE STRPOS(LOWER(IFNULL(title_product,'')), a) > 0
        )) AS acc_title_hits,
        IF(
          LOWER(IFNULL(@brand_focus,'')) != '' AND (
            STRPOS(LOWER(IFNULL(brand,'')), LOWER(IFNULL(@brand_focus,''))) > 0 OR
            STRPOS(LOWER(IFNULL(title_product,'')), LOWER(IFNULL(@brand_focus,''))) > 0
          ),
          1, 0
        ) AS matches_brand_relaxed,
        IF(
          LOWER(IFNULL(@brand_focus,'')) != '' AND LOWER(IFNULL(brand,'')) = LOWER(IFNULL(@brand_focus,'')),
          1, 0
        ) AS matches_brand_strict,

        IF(REGEXP_CONTAINS(LOWER(IFNULL(title_product,'')), r'\\bipad\\b'),1,0) AS is_ipad,
        IF(REGEXP_CONTAINS(LOWER(IFNULL(title_product,'')), r'\\btablet\\b'),1,0) AS is_tablet,
        IF(REGEXP_CONTAINS(LOWER(IFNULL(title_product,'')), r'\\bsmartwatch\\b|\\breloj\\s+inteligente\\b|\\bwatch\\b'),1,0) AS is_smartwatch,
        IF(REGEXP_CONTAINS(LOWER(IFNULL(title_product,'')), r'power\\s*bank|bateria\\s*externa|batería\\s*externa'),1,0) AS is_powerbank,
        IF(REGEXP_CONTAINS(LOWER(IFNULL(title_product,'')), r'cargador\\s*inalambr|cargador\\s*inalámbr|\\bcargador\\b'),1,0) AS is_charg_wireless,
        IF(REGEXP_CONTAINS(LOWER(IFNULL(title_product,'')), r'audifono|audífono|audifonos|audífonos|earbuds|airpods|earpods|auriculares'),1,0) AS is_headphones
      FROM base
    ),
    filtered AS (
      SELECT * FROM enriched
      WHERE es_celular = 0
        AND acc_title_hits > 0
        AND (
          @require_brand = FALSE
          OR (@strict_brand = TRUE  AND matches_brand_strict  = 1)
          OR (@strict_brand = FALSE AND matches_brand_relaxed = 1)
        )
        AND (
          @cat_hint = '' OR
          (@cat_hint = 'ipad'           AND is_ipad = 1) OR
          (@cat_hint = 'tablet'         AND (is_tablet = 1 OR is_ipad = 1)) OR
          (@cat_hint = 'smartwatch'     AND is_smartwatch = 1) OR
          (@cat_hint = 'powerbank'      AND is_powerbank = 1) OR
          (@cat_hint = 'charg_wireless' AND is_charg_wireless = 1) OR
          (@cat_hint = 'headphones'     AND is_headphones = 1)
        )
    ),
    scored AS (
      SELECT
        * ,
        (
          5 * CASE WHEN @strict_brand = TRUE THEN matches_brand_strict ELSE matches_brand_relaxed END +
          2 * acc_title_hits +
          1 * ARRAY_LENGTH(ARRAY(
                SELECT t FROM UNNEST(@terms) AS t WHERE STRPOS(LOWER(IFNULL(title_product,'')), t) > 0
          ))
        ) AS match_score
      FROM filtered
    )
    SELECT *
    FROM scored
    ORDER BY
      CASE WHEN @strict_brand = TRUE THEN matches_brand_strict ELSE matches_brand_relaxed END DESC,
      match_score DESC,
      COALESCE(porcentaje_descuento, 0) DESC
    LIMIT 50
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("terms", "STRING", nq_terms),
            bigquery.ArrayQueryParameter("acc", "STRING", [t.lower() for t in ACCESSORY_TERMS]),
            bigquery.ScalarQueryParameter("brand_focus", "STRING", brand_focus or ""),
            bigquery.ScalarQueryParameter("require_brand", "BOOL", bool(brand_focus) and require_brand),
            bigquery.ScalarQueryParameter("strict_brand", "BOOL", bool(brand_focus) and strict_brand),
            bigquery.ScalarQueryParameter("cat_hint", "STRING", cat_hint or ""),
        ]
    )
    rows = list(bq_client.query(sql, job_config=job_config).result())

    def to_item(r):
        return {
            "id": r["id"], "marca": r["brand"], 
            "producto": r["title_product"][:38],
            "imagen": r["image"],
            "url": r["calltoaction_url"], "tipo": r["line"] or "Producto",
            "plan": r["title_plan"], "price_list": r["price_list"], "discount": r["discount"],
            "porcentaje_descuento": r["porcentaje_descuento"], "line": r["line"],
            "modality": r["modality"], "financing": r["financing"],
        }
    return [to_item(r) for r in rows]

def search_accessories_bq(nq: str) -> List[dict]:
    terms = _terms_from_query(nq)
    if not terms:
        return []
    brand_focus = _canonical_brand_from_query(nq)
    cat_hint = _category_hint_from_query(nq)

    unique_items = {}
    
    if brand_focus:
        res_brand = _run_accessories_query(terms, brand_focus=brand_focus, require_brand=True, strict_brand=False, cat_hint=cat_hint)
        for item in res_brand:
            unique_items.setdefault(item['id'], item)

    if cat_hint:
        res_generic_cat = _run_accessories_query(terms, brand_focus=None, require_brand=False, strict_brand=False, cat_hint=cat_hint)
        for item in res_generic_cat:
            unique_items.setdefault(item['id'], item)

    if len(unique_items) < 5 and not brand_focus and 'cargador' in nq.lower():
         res_wide = _run_accessories_query(terms, brand_focus=None, require_brand=False, strict_brand=False, cat_hint='charg_wireless')
         for item in res_wide:
            unique_items.setdefault(item['id'], item)

    final_list = list(unique_items.values())
    final_list.sort(key=lambda x: (x.get('match_score', 0) if isinstance(x, dict) else 0, x.get('porcentaje_descuento') or 0), reverse=True)
    
    result_final, used_ids, selected_categories = [], set(), set()
    ACCESSORY_TYPE_MAPPER = {
        'headphones': ['audifonos', 'earbuds', 'airpods', 'earpods','auriculares'],
        'charg_wireless': ['cargador inalambrico', 'cargador inalámbrico', 'cargador'],
        'powerbank': ['power bank', 'bateria externa', 'batería externa'],
        'case_funda': ['funda', 'case', 'protector'],
        'cable_adaptador': ['cable', 'adaptador'],
        'ipad': ['ipad'],
        'tablet': ['tablet'],
        'smartwatch': ['watch', 'reloj'],
    }
    def get_accessory_category(item: dict) -> str:
        title = item['producto'].lower()
        for cat, keywords in ACCESSORY_TYPE_MAPPER.items():
            if any(k in title for k in keywords):
                return cat
        return 'other'

    for item in final_list:
        if len(result_final) >= 5:
            break
        item_cat = get_accessory_category(item)
        if item['id'] in used_ids:
            continue
        if item_cat not in selected_categories or (brand_focus and item['marca'].lower() == brand_focus and item['marca'].lower() not in selected_categories):
             result_final.append(item)
             used_ids.add(item['id'])
             selected_categories.add(item_cat)

    if len(result_final) < 5:
        remaining_pool = sorted([item for item in final_list if item['id'] not in used_ids], key=lambda x: x.get('price_list') or 1e9)
        result_final.extend(remaining_pool[:5 - len(result_final)])

    return result_final[:5]

# ------------------ BQ: Planes ------------------
def search_plans_bq(nq: str) -> List[dict]:
    is_prepago = _has_prepago_term(nq) or 'prepago' in nq.lower()
    line_filter_product = "Claro Chip Claro Prepago" if is_prepago else "Claro Chip Claro Postpago"
    line_label = "Postpago" if not is_prepago else "Prepago"
    
    sql = f"""
    SELECT
      id, brand, title_product, image, calltoaction_url, line, modality, financing,
      title_plan,
      SAFE_CAST(NULLIF(CAST(price_list AS STRING), '') AS FLOAT64) AS price_list,
      SAFE_CAST(NULLIF(CAST(discount AS STRING), '') AS FLOAT64)   AS discount,
      CASE
          WHEN price_list IS NOT NULL AND price_list > 0 AND discount IS NOT NULL
            THEN GREATEST(0, LEAST(100, ROUND(((price_list - discount) / price_list) * 100)))
          ELSE NULL
      END AS porcentaje_descuento
    FROM `{BQ_TABLE_PRODUCTS}`
    WHERE 
      title_product LIKE @line_filter_product
      AND LOWER(IFNULL(line,'')) = LOWER(@line_label)
    ORDER BY
      CASE 
        WHEN REGEXP_CONTAINS(LOWER(IFNULL(modality,'')), r'(portabilidad)') THEN 0
        WHEN REGEXP_CONTAINS(LOWER(IFNULL(modality,'')), r'(linea nueva|línea nueva)') THEN 1
        ELSE 99 
      END,
      COALESCE(price_list, 9e18) ASC
    LIMIT 10
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("line_filter_product", "STRING", f"%{line_filter_product}%"),
            bigquery.ScalarQueryParameter("line_label", "STRING", line_label),
        ]
    )
    rows = list(bq_client.query(sql, job_config=job_config).result())

    plans = [{
        "id": r["id"], "marca": r["brand"], 
        "producto": r["title_product"][:38],
        "imagen": r["image"],
        "url": r["calltoaction_url"], "tipo": line_label,
        "plan": r["title_plan"], "price_list": r["price_list"], "discount": r["discount"],
        "porcentaje_descuento": r["porcentaje_descuento"], "line": r["line"], "modality": r["modality"], "financing": r["financing"],
    } for r in rows]
    
    return plans

# ------------------ Recomendados fijos ------------------
DEFAULT_RECOMMENDATIONS = [
    {"nombre": "Celulares en Tienda Claro", "texto": "Explora todos los equipos, ofertas y promociones. Compra online y recíbelo en casa.", "url": "https://www.claro.com.pe/personas/tienda/celulares/"},
    {"nombre": "Planes Postpago Claro", "texto": "Elige tu Plan Max Ilimitado y obtén beneficios para tu nuevo equipo.", "url": "https://www.claro.com.pe/personas/movil/postpago/"},
    {"nombre": "Móvil Claro", "texto": "Información de prepago, postpago y servicios móviles en un solo lugar.", "url": "https://www.claro.com.pe/personas/movil/"}
]

# ------------------ Claro Video (Optimizado) ------------------
def build_claro_video_response(normalized_query: str) -> dict:
    try:
        main_movies, related_movies = search_claro_video_catalog(normalized_query)
    except Exception as e:
        logger.exception("Error construyendo respuesta de Claro video", exc_info=e)
        main_movies, related_movies = [], []

    if not main_movies:
        descripcion = (
            "Por el momento no encontré una película específica en el catálogo de Claro video "
            f"para tu búsqueda «{normalized_query}», pero puedes explorar las opciones "
            "de películas, series y TV en vivo desde la web o app de Claro video."
        )

        listado = [
            {
                "nombre": "Claro video: Películas, series y TV en vivo",
                "texto": (
                    "Disfruta estrenos, clásicos y contenido para toda la familia. "
                    "Accede desde tu móvil, tablet, Smart TV o web y alquila o compra tus títulos favoritos."
                ),
                "url": "https://www.clarovideo.com/peru/home",
                "imagen": None,
            }
        ]

        relacionados = [
            {
                "nombre": "Catálogo de películas",
                "texto": "Explora el catálogo completo de películas en Claro video.",
                "url": "https://www.clarovideo.com/peru/categoria/peliculas",
                "imagen": None,
            },
            {
                "nombre": "Series y temporadas completas",
                "texto": "Encuentra tus series favoritas y mira temporadas completas en streaming.",
                "url": "https://www.clarovideo.com/peru/categoria/series",
                "imagen": None,
            },
        ]

        return {
            "status": "NotFound",
            "query": normalized_query,
            "tipo": "pelicula",
            "descripcion": descripcion,
            "listado": listado,
            "relacionados": relacionados,
        }

    first_movie = main_movies[0]
    titulo_pelicula = (first_movie.get("title") or first_movie.get("title_original") or "esta película")
    titulo_marketero = f"¡Disfruta «{titulo_pelicula}» en Claro Video!"

    listado = []
    for m in main_movies:
        desc = (m.get("description_large") or m.get("description") or "")
        # [OPTIMIZACIÓN] Truncar descripción a 200 chars
        if len(desc) > 150: desc = desc[:150] + "..."
        listado.append({
            "nombre": m.get("title") or m.get("title_original"),
            "texto": desc,
            "url": m.get("url"),
            "imagen": (m.get("image_medium") or m.get("image_small") or m.get("image_large")),
        })

    relacionados = []
    for m in related_movies:
        desc = (m.get("description_large") or m.get("description") or "")
        # [OPTIMIZACIÓN] Truncar descripción a 200 chars
        if len(desc) > 150: desc = desc[:150] + "..."
        relacionados.append({
            "nombre": m.get("title") or m.get("title_original"),
            "texto": desc,
            "url": m.get("url"),
            "imagen": (m.get("image_medium") or m.get("image_small") or m.get("image_large")),
        })

    descripcion = (
        "Accede a Claro video con tus credenciales de Claro y disfruta de estas películas cuando quieras. "
        "Además, te mostramos otras opciones relacionadas que también podrían interesarte."
    )

    return {
        "status": "Found",
        "query": normalized_query,
        "tipo": "pelicula",
        "titulo": titulo_marketero,
        "descripcion": descripcion,
        "listado": listado,
        "relacionados": relacionados,
    }

# ------------------ Endpoint ------------------
@dev_bp.route("/query", methods=["GET"])
@limiter.limit("10 per minute")
def query_dev():
    user_query = request.args.get("q", "")
    sia_id = request.args.get("sia_id", "")
    user_agent = request.headers.get("User-Agent", "")
    pregunta_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

#    cors_resp = origin_check(
#        request,
#        allowed=[
#            "https://www.claro.com.pe",
#            "https://test-claro-pe.prod.clarodigital.net",
#            "https://search-test-1079186964678.us-central1.run.app",
#            "https://search-api-1079186964678.us-central1.run.app",
#            "https://genia-front-test-1079186964678.us-central1.run.app",
#        ],
#    )
#    if cors_resp:
#        return cors_resp
    if os.environ.get("DEV_ORIGIN_CHECK", "false").lower() == "true":
        cors_resp = origin_check(request, allowed=DEV_ALLOWED_ORIGINS)
        if cors_resp:
            return cors_resp


    if not user_query:
        return jsonify({"error": "El parámetro 'q' es requerido"}), 400

    user_query = sanitize_query(user_query)

    try:
        normalized_query, category = classify_with_gemini(user_query)
        if category == "celulares":
            normalized_query = _strip_fillers_from_query(normalized_query)

        # category = clasificador(normalized_query)


        if category == "pelicula":
          try:
              respuesta = call_clarovideo_movies_api(normalized_query)
          except Exception as e:
              logger.exception("Error llamando ClaroVideo Movies API", exc_info=e)
              respuesta = {
                  "_meta": {"cache_hit": False, "category": "pelicula", "normalized_query": normalized_query},
                  "descripcion": "Ocurrió un error al buscar en Claro video. Intenta nuevamente más tarde.",
                  "listado": [],
                  "relacionados": [],
                  "query": normalized_query,
                  "status": "Error",
                  "tipo": "pelicula",
                  "tipo_respuesta": "general",
                  "titulo": "Claro video: búsqueda de películas"
              }

          # Si quieres que tu backend “mande” el meta siempre (aunque el microservicio ya lo tenga)
          respuesta["tipo_respuesta"] = respuesta.get("tipo_respuesta", "general")
          respuesta["_meta"] = {"normalized_query": normalized_query, "category": category, "cache_hit": False}

          guardar_pregunta_en_historial(normalized_query, sia_id, respuesta, pregunta_timestamp, user_agent)
          return jsonify(respuesta)


        respuesta = None
        uso_cache = False
        
        # [FIX CRÍTICO]: brand_name definido antes
        brand_name = _canonical_brand_from_query(normalized_query)
        display_phone_query = _display_query_for_phone(normalized_query) 

        if category == "general":
            cache = buscar_respuesta_definitiva(normalized_query)
            if cache:
                uso_cache = True
                respuesta = json.loads(cache)
        
        # [FIX CRÍTICO]: Cache para Planes
        if respuesta is None and category == "planes":
            planes = search_plans_bq(normalized_query)
            plan_type = "Postpago" if not _has_prepago_term(normalized_query) and 'postpago' in normalized_query.lower() else "Prepago"
            general_response = get_summary_from_vertex(normalized_query)

            respuesta = {
                "titulo": general_response.get("titulo", f"Planes Móviles {plan_type} Claro"),
                "descripcion": general_response.get("descripcion", f"¡Descubre la libertad y beneficios de nuestros planes {plan_type} Claro!"),
                "planes": planes,
                "listado": general_response.get("listado", []),
                "relacionados": general_response.get("relacionados", []),
                "status": "Found" if planes or general_response.get("status") == "Found" else "Not Found",
                "query": normalized_query,
                "tipo_respuesta": "general",
            }
            
            respuesta["_meta"] = {"normalized_query": normalized_query, "category": category, "cache_hit": uso_cache}
            guardar_pregunta_en_historial(normalized_query, sia_id, respuesta, pregunta_timestamp, user_agent)
            return jsonify(respuesta)

        if respuesta is None and category == "tienda_productos":
            producto_tp = search_accessories_bq(normalized_query)
            if brand_name == "lenovo":
                respuesta = {
                    "titulo": f"¡Ofertas Exclusivas Lenovo en Tienda Claro!",
                    "descripcion": f"Descubre lo que tenemos para ti: ¡lo último en tecnología te espera!",
                    "producto": producto_tp,
                    "recomendados": DEFAULT_RECOMMENDATIONS,
                    "status": "Found" if producto_tp else "Not Found",
                    "tipo_respuesta": "tienda",
                }
            elif producto_tp:
                respuesta = {
                    "titulo": f"¡Encontramos {normalized_query.title()} para ti!",
                    "descripcion": "¡Potencia tu mundo con la mejor tecnología! Mira estas opciones irresistibles.",
                    "producto": producto_tp,
                    "recomendados": DEFAULT_RECOMMENDATIONS,
                    "status": "Found",
                    "tipo_respuesta": "tienda",
                }
            else:
                respuesta = {
                    "titulo": f"Ups, no encontramos '{normalized_query}' por ahora",
                    "descripcion": "No encontramos coincidencias exactas en accesorios. Explora nuestras recomendaciones más populares:",
                    "producto": [],
                    "recomendados": DEFAULT_RECOMMENDATIONS,
                    "status": "Not Found",
                    "tipo_respuesta": "tienda",
                }
            respuesta["_meta"] = {"normalized_query": normalized_query, "category": category, "cache_hit": uso_cache}
            guardar_pregunta_en_historial(normalized_query, sia_id, respuesta, pregunta_timestamp, user_agent)
            return jsonify(respuesta)
        
        is_simple_brand_query = normalized_query in ["iphone", "samsung", "galaxy", "xiaomi", "redmi", "poco", "motorola", "moto"]
        only_brand_corrected = (
            brand_name is not None
            and len(_nums_from_query(normalized_query)) == 0
            and (is_simple_brand_query or not _match_any(PHONE_MODEL_PATTERNS, normalized_query))
        )
        
        if respuesta is None:
            if category == "celulares" and only_brand_corrected:
                brand_items = search_brand_only_bq(normalized_query)
                if brand_items:
                    accesorios = search_accessories_bq(normalized_query)
                    #brand_cap = _canonical_brand_from_query(normalized_query).capitalize()
                    brand_cap = _brand_display_label(normalized_query) or display_phone_query
                    brand_desc = _pick_brand_description(brand_cap, normalized_query)
                    respuesta = {
                        "titulo": f"Lo mas vendido de {brand_cap} !😍",
                        "descripcion": brand_desc,
                        "producto": brand_items,
                        "recomendados": accesorios if accesorios else DEFAULT_RECOMMENDATIONS,
                        "status": "Found",
                        "tipo_respuesta": "tienda",
                    }
                    respuesta["_meta"] = {"normalized_query": normalized_query, "category": category, "cache_hit": uso_cache}
                    guardar_pregunta_en_historial(normalized_query, sia_id, respuesta, pregunta_timestamp, user_agent)
                    return jsonify(respuesta)

            if category == "celulares":
                topN, had_exact, wanted_label = search_products_bq(
                    normalized_query,
                    prioritize_plan_79=(category == "celulares")
                )
                if brand_name:
                    if not topN:
                        topN = search_brand_only_bq(normalized_query)
                    elif len(topN) < 5:
                        brand_fallback = search_brand_only_bq(normalized_query)
                        if brand_fallback:
                            existing_ids = {item.get("id") for item in topN}
                            for item in brand_fallback:
                                if len(topN) >= 5:
                                    break
                                if item.get("id") in existing_ids:
                                    continue
                                topN.append(item)
                                existing_ids.add(item.get("id"))
                accesorios = search_accessories_bq(normalized_query)
                recomendados = accesorios if accesorios else DEFAULT_RECOMMENDATIONS
                
                if topN:
                    if had_exact:
                        #desc = f"¡Lo tenemos! El {wanted_label.title()} te espera con ofertas exclusivas en planes de renovación y portabilidad." 
                        #desc = f"🔥 ¡Buenísima elección! Sí tenemos el {wanted_label} en Tienda Claro, te espera con ofertas exclusivas en planes de renovación y portabilidad."                     
                        desc = _pick_phone_desc(wanted_label, normalized_query, PHONE_DESC_EXACT_TEMPLATES)                    
                    else:
                        #desc = f"¡Te mostramos opciones irresistibles! No tenemos el {wanted_label.title()} exacto, pero estos modelos similares te encantarán."
                        #desc = f"¡Te mostramos opciones irresistibles! No tenemos el {wanted_label} exacto, pero estos modelos similares te encantarán."  
                        desc = _pick_phone_desc(wanted_label, normalized_query, PHONE_DESC_SIMILAR_TEMPLATES) 

                    title_templates = PHONE_TITLE_EXACT_TEMPLATES if had_exact else PHONE_TITLE_SIMILAR_TEMPLATES                   
                    respuesta = {
                        "titulo": _pick_phone_title(wanted_label, normalized_query, title_templates),
                        "descripcion": desc,
                        "producto": topN,
                        "recomendados": recomendados,
                        "status": "Found",
                        "tipo_respuesta": "tienda",
                    }
                else:
                    respuesta = {
#                        "titulo": f"Ups, no encontramos el modelo {normalized_query.title()}",
#                        "descripcion": f"No encontramos ese modelo exacto, pero mira estas alternativas increíbles o explora nuestras recomendaciones:",
                        "titulo": f"Ups, no encontramos el modelo {display_phone_query}",
                        "descripcion": _pick_phone_desc(display_phone_query, normalized_query, PHONE_DESC_NOT_FOUND_TEMPLATES),
                        "producto": [],
                        "recomendados": recomendados,
                        "status": "Not Found",
                        "tipo_respuesta": "tienda",
                    }
            else:
                respuesta = get_summary_from_vertex(normalized_query)
                respuesta["query"] = normalized_query
                respuesta["tipo_respuesta"] = "general"
                guardar_en_respuestas_definitivas(normalized_query, respuesta)

        guardar_pregunta_en_historial(normalized_query, sia_id, respuesta, pregunta_timestamp, user_agent)
        respuesta["_meta"] = {"normalized_query": normalized_query, "category": category, "cache_hit": uso_cache}

# ✅ FALLBACK A PELÍCULAS (ÚNICO PUNTO)
        if category == "general" and respuesta:
            status = respuesta.get("status")
            is_not_found = status in ("NotFound", "Not Found")
            is_clarovideo_disfruta = title_is_clarovideo_disfruta(respuesta)
            is_clarovideo_descripcion = description_is_clarovideo_disfruta(respuesta)
            is_clarovideo_pelicula = description_is_clarovideo_genero(respuesta) 

            if is_not_found or is_clarovideo_disfruta or is_clarovideo_descripcion or is_clarovideo_pelicula:
                movies_hit = try_movies_first(normalized_query)
                if movies_hit:
                    movies_hit["_meta"] = {
                        "cache_hit": False,
                        "category": "pelicula",
                        "normalized_query": normalized_query,
                    }
                    guardar_pregunta_en_historial(
                        normalized_query,
                        sia_id,
                        movies_hit,
                        pregunta_timestamp,
                        user_agent
                    )
                    return jsonify(movies_hit)


        # ✅ Si no hubo fallback, recién guardas el historial normal
        guardar_pregunta_en_historial(normalized_query, sia_id, respuesta, pregunta_timestamp, user_agent)
        return jsonify(respuesta)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[dev] Error procesando '{user_query}': {error_msg}", exc_info=True)
        return jsonify({
            "titulo": "Error interno del sistema",
            "descripcion": f"<p>Ocurrió un inconveniente al procesar tu solicitud. ({error_msg})</p><p>Por favor intenta nuevamente en unos segundos.</p>",
            "query": user_query,
            "status": "Error",
            "tipo_respuesta": "general",
            "error_detalle": error_msg
        }), 500