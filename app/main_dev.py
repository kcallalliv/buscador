# app/main_dev.py
import datetime
import json
import logging
import os
import re
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

# ------------------ BQ ------------------
bq_client = bigquery.Client()
BQ_TABLE_PRODUCTS = "prd-claro-mktg-data-storage.tienda_claro.productos"
BQ_TABLE_TOP5 = "prd-claro-mktg-data-storage.master_analytics.sales_top5_brand_monthly"
BQ_TABLE_CLAROVIDEO = "prd-claro-mktg-data-storage.claro_video.catalogo" 

# ------------------ Blueprint ------------------
dev_bp = Blueprint("dev_bp", __name__)
CORS(dev_bp, resources={r"/dev/*": {"origins": [
    "https://www.claro.com.pe",
    "https://test-claro-pe.prod.clarodigital.net",
    "https://search-test-1079186964678.us-central1.run.app",
]}})

# ------------------ Helpers Vertex ------------------
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
    # Reemplazos conservadores (palabra completa)
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

# Stoplist global para leaks cuando brand_lock está activo
# Términos de otras marcas que NO deben aparecer cuando hay brand_lock
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

# Modelos
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

def search_claro_video_catalog(
    normalized_query: str,
    max_main: int = 8,
    max_related: int = 12,
):
    """
    Busca en el catálogo de Claro video usando BigQuery, y devuelve:
      - main_movies: lista de películas principales (todas las coincidencias fuertes)
      - related_movies: lista de películas relacionadas para rellenar

    Enfoque robusto:
      - NO filtramos por términos en el SQL (evitamos perdernos títulos raros).
      - Traemos un subconjunto grande del catálogo (LIMIT).
      - Ordenamos por similitud (rapidfuzz.WRatio) entre la query y:
          * title
          * title_original
          * title_uri (slug con guiones reemplazados por espacios)
    """

    nq = normalize_query_local(normalized_query)
    nq_base = strip_accents((nq or "").lower()).strip()
    if not nq_base:
        return [], []

    # 1) Trae un subconjunto amplio del catálogo
    sql = f"""
    SELECT
      id,
      section,
      title,
      title_original,
      year,
      duration,
      description,
      description_large,
      rating_code,
      title_uri,
      url,
      image_small,
      image_medium,
      image_large
    FROM `{BQ_TABLE_CLAROVIDEO}`
    -- Si tienes columna para tipo de contenido, aquí puedes limitar solo películas:
    -- WHERE LOWER(IFNULL(section, '')) = 'pelicula'
    LIMIT 2000
    """

    try:
        rows = list(bq_client.query(sql).result())
    except Exception as e:
        logger.exception("Error consultando catálogo Claro video", exc_info=e)
        return [], []

    if not rows:
        return [], []

    # 2) Fuzzy matching sobre título, título original y slug
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

    if not scored:
        return [], []

    # Ordenar por mejor score
    scored.sort(key=lambda x: x[0], reverse=True)

    best_score = scored[0][0]
    # Si el mejor score es muy bajo, consideramos que no hay match razonable
    if best_score < 40:
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

    main_movies: List[dict] = []
    related_movies: List[dict] = []

    # 3) MAIN: coincidencias directas (query como substring) con buen score
    for score, row, title, title_orig, slug in scored:
        if score < 40:
            break

        if (
            nq_base in title
            or nq_base in title_orig
            or nq_base in slug
        ):
            main_movies.append(row_to_movie(row))

        if len(main_movies) >= max_main:
            break

    # 4) Si no hubo suficientes direct hits, rellenar main_movies con los mejores scores
    if not main_movies:
        for score, row, *_ in scored:
            if score < 40:
                break
            main_movies.append(row_to_movie(row))
            if len(main_movies) >= max_main:
                break

    used_ids = {m["id"] for m in main_movies}

    # 5) RELATED: siguientes mejores que no estén en main_movies
    for score, row, *_ in scored:
        if len(related_movies) >= max_related:
            break
        if row.id in used_ids:
            continue
        if score < 30:
            continue
        related_movies.append(row_to_movie(row))

    return main_movies, related_movies


PLAN_HINTS = {
    "plan", "planes", "postpago", "prepago", "pospago", "linea nueva", "línea nueva", "portabilidad", "renovacion", "renovación",
    "chip", "sim card", "tarjeta sim"
}

def _match_any(patterns, text: str) -> bool:
    return any(p.search(text) for p in patterns)

GEMINI_INTENT_PROMPT = """
Eres un clasificador de consultas para el buscador de Claro Perú.
Tu tarea: (1) corregir/estandarizar la consulta; (2) clasificarla en una categoría.
Categorías:
- "celulares": equipos/marcas/modelos de teléfonos (iphone, samsung galaxy, xiaomi/redmi/poco, motorola, huawei/honor, etc.).
- "tienda_productos": catálogo de Tienda Claro NO celulares (routers, decodificadores, planes hogar/negocios fijos, accesorios como power bank/cargadores/audífonos/smartwatch, etc.).
- "planes": cuando la intención es buscar planes móviles (postpago, prepago, portabilidad, renovación) o solo el chip.
- "pelicula": cuando la intención del usuario es el nombre de una película o franquicia.
- "general": preguntas informativas, soporte, trámites u otros no cubiertos.
Responde SOLO JSON:
{ "normalized_query": "<minúsculas y corregida>", "category": "celulares" | "tienda_productos" | "planes" | "pelicula" | "general" }
"""

def classify_with_gemini(user_query: str) -> Tuple[str, str]:
    nq_local = normalize_query_local(user_query)
    try:
        payload = {"query": user_query}
        result = call_gemini_json(system_prompt=GEMINI_INTENT_PROMPT, user_input=json.dumps(payload))
        model_norm = normalize_query_local(result.get("normalized_query") or nq_local)
        normalized = _safe_pick_normalized(model_norm, nq_local)
        category = (result.get("category") or "general").strip().lower()
    except Exception:
        normalized = nq_local
        category = "general"

    # Overrides heurísticos
    has_phone_model = _match_any(PHONE_MODEL_PATTERNS, normalized)
    terms = set(normalized.split())
    has_phone_brand = bool(terms & BRAND_HINTS)
    txt = f" {normalized} "
    has_accessory = any(f" {kw} " in txt for kw in ACCESSORY_TERMS) or ('airpods' in normalized) or ('earpods' in normalized)
    has_store = any(f" {kw} " in txt for kw in STORE_SIGNALS)
    generic_phone_words = any(w in normalized for w in ["celular","smartphone","equipo","postpago","prepago","liberado","libre"])
    is_movie = any(h in normalized for h in MOVIE_HINTS)
    is_plan_query = bool(terms & PLAN_HINTS) and not (has_phone_model or has_phone_brand or has_accessory)
    
    # Prioridad: Película > Plan > Celular > Tienda Producto > General
    
    if is_movie:
        category = "pelicula"
    elif is_plan_query and not has_phone_model:
        category = "planes"
    elif has_phone_model:
        category = "celulares"
    elif has_store or has_accessory:
        category = "tienda_productos"
    elif has_phone_brand and generic_phone_words:
        category = "celulares"
    elif has_phone_brand:
        category = "celulares"
    # Si detecta plan y no es un modelo específico, forzar a 'planes'
    elif is_plan_query:
        category = "planes"

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

BRAND_TERMS_FOR_MODEL = {
    "iphone","samsung","galaxy","xiaomi","redmi","poco","huawei",
    "motorola","moto","pixel","honor","infinix","tecno","oppo","realme","vivo","lenovo","zte"
}

def _brand_terms_from_query(nq: str) -> List[str]:
    terms = set(_terms_from_query(nq))
    return sorted(list(terms & BRAND_TERMS_FOR_MODEL))

def _canonical_brand_from_query(nq: str) -> Optional[str]:
    s = nq.lower()
    # Apple
    if "iphone" in s or "apple" in s:
        return "apple"
    # Samsung
    if "samsung" in s or "galaxy" in s:
        return "samsung"
    # Xiaomi ecosistema: redmi/poco => canonical xiaomi
    if "xiaomi" in s or "redmi" in s or re.search(r"\bpoco\b", s):
        return "xiaomi"
    # Resto
    if "huawei" in s:                 return "huawei"
    if "motorola" in s or re.search(r"\bmoto\b", s): return "motorola"
    if "pixel" in s or "google" in s: return "google"
    if "honor" in s:                  return "honor"
    if "infinix" in s:                return "infinix"
    if "tecno" in s:                  return "tecno"
    if "oppo" in s:                   return "oppo"
    if "realme" in s:                 return "realme"
    if "vivo" in s:                   return "vivo"
    if "lenovo" in s:                 return "lenovo"
    if "zte" in s:                    return "zte"
    return None


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
    
    if len(final_items) < 5:
        remaining_pool = [r for r in pool_sorted if r['id'] not in selected_ids]
        final_items.extend(remaining_pool[:5 - len(final_items)])
        
    return final_items[:5]

def _category_hint_from_query(nq: str) -> str:
    s = nq.lower()
    if "ipad" in s: return "ipad"
    if "tablet" in s: return "tablet"
    if "smartwatch" in s or "reloj inteligente" in s or "watch" in s: return "smartwatch"
    if "power bank" in s or "powerbank" in s or "bateria externa" in s or "batería externa" in s: return "powerbank"
    if "cargador" in s and ("inalambr" in s or "inalámbr" in s): return "charg_wireless"
    if "audifono" in s or "audifonos" in s or "audífono" in s or "audífonos" in s or "earbuds" in s or "airpods" in s or "earpods" in s or "auriculares" in s: return "headphones"
    return ""

# ------------------ BQ: Productos (consulta general) ------------------
def search_products_bq(nq: str, prioritize_plan_79: bool):
    terms = _terms_from_query(nq)
    if not terms: return [], False, nq

    want_prepago = _has_prepago_term(nq)
    explicit_mod = _explicit_modality_from_query(nq)
    storage = _storage_from_query(nq) or ""
    cheap = _query_mentions_economy(nq)
    qnum = _extract_main_number(nq)
    series = _samsung_series_from_query(nq)

    canon = _canonical_brand_from_query(nq) or ""
    brand_aliases, brand_kw = [], ""
    if canon == "apple":
        brand_aliases, brand_kw = ["apple","iphone"], "iphone"
    elif canon == "samsung":
        brand_aliases, brand_kw = ["samsung","galaxy","samsung mobile","samsung electronics"], "galaxy"
    elif canon == "motorola":
        brand_aliases, brand_kw = ["motorola","moto"], "moto"
    elif canon == "xiaomi":
        # Redmi y POCO se tratan como ecosistema Xiaomi
        brand_aliases, brand_kw = ["xiaomi","redmi","poco"], "xiaomi"
    elif canon == "google":
        brand_aliases, brand_kw = ["google","pixel"], "pixel"
    elif canon in ("huawei","honor","oppo","realme","vivo","infinix","tecno","lenovo","zte","tcl","nokia"):
        brand_aliases, brand_kw = [canon], canon

    brand_lock = bool(canon)  # si hay marca canónica, activamos lock
    other_terms = OTHER_BRAND_TERMS.get(canon, [])


    force_prepago = want_prepago

    # Flags de variantes según la consulta
    want_pro_max = 1 if re.search(r'\bpro\s*max\b', nq, re.I) else 0
    want_pro     = 1 if (re.search(r'\bpro\b', nq, re.I) and not want_pro_max) else 0
    want_ultra   = 1 if re.search(r'\bultra\b', nq, re.I) else 0
    want_plus    = 1 if re.search(r'(\bplus\b|\b\+\b)', nq, re.I) else 0
    want_lite    = 1 if re.search(r'\blite\b', nq, re.I) else 0
    want_fe      = 1 if re.search(r'\bfe\b', nq, re.I) else 0
    want_mini    = 1 if re.search(r'\bmini\b', nq, re.I) else 0
    want_se      = 1 if re.search(r'\bse\b', nq, re.I) else 0

    # SOLUCIÓN AL ERROR DE AMBITO: Se unifican scored_stage y scored
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

    /* Guardas de marca + stoplist */
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
    ),

    /* Normalización: colores y capacidad (como en TOP) */
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

    /* Tokens exactos/variantes */
    enriched AS (
      SELECT
        enriched_base2.*,
        
        -- iPhone tokens (con exclusiones para modelo base)
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

        -- iPhone variantes
        IF(REGEXP_CONTAINS(tp_lc, r'\\bmini\\b'),1,0) AS has_mini,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bpro[[:space:]]*max\\b'),1,0) AS has_pro_max,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bpro\\b') AND NOT REGEXP_CONTAINS(tp_lc, r'\\bmax\\b'),1,0) AS has_pro_only,
        IF(REGEXP_CONTAINS(tp_lc, r'\\b\\+\\b|\\bplus\\b'),1,0) AS has_plus,
        IF(REGEXP_CONTAINS(tp_lc, r'\\blite\\b'),1,0) AS has_lite,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bfe\\b'),1,0) AS has_fe,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bse\\b'),1,0) AS has_se,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bultra\\b'),1,0) AS has_ultra,

        -- Samsung S series (con exclusiones para base)
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*s[[:space:]]*25[[:space:]]*ultra\\b'),1,0) AS tok_s25ultra,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*s[[:space:]]*25[[:space:]]*\\+\\b|\\bgalaxy\\s*s\\s*25\\s*plus\\b'),1,0) AS tok_s25plus,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*s[[:space:]]*25[[:space:]]*fe\\b'),1,0) AS tok_s25fe,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*s[[:space:]]*25\\b') AND NOT REGEXP_CONTAINS(tp_lc, r'\\b(ultra|plus|fe)\\b'),1,0) AS tok_s25,

        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*s[[:space:]]*24[[:space:]]*ultra\\b'),1,0) AS tok_s24ultra,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*s[[:space:]]*24[[:space:]]*\\+\\b|\\bgalaxy\\s*s\\s*24\\s*plus\\b'),1,0) AS tok_s24plus,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*s[[:space:]]*24\\b') AND NOT REGEXP_CONTAINS(tp_lc, r'\\b(ultra|plus|fe)\\b'),1,0) AS tok_s24,

        -- Samsung A series (A55, A35, etc.)
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*a[[:space:]]*56\\b'),1,0) AS tok_a56,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*a[[:space:]]*55\\b'),1,0) AS tok_a55,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*a[[:space:]]*36\\b'),1,0) AS tok_a36,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*a[[:space:]]*35\\b'),1,0) AS tok_a35,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*a[[:space:]]*17\\b'),1,0) AS tok_a17,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*a[[:space:]]*16\\b'),1,0) AS tok_a16,
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*a[[:space:]]*15\\b'),1,0) AS tok_a15,

        -- genéricos familia+número
        IF(REGEXP_CONTAINS(tp_lc, r'\\bgalaxy[[:space:]]*(s|a|m)[[:space:]]*[0-9]{1,2}\\b'), 1, 0) AS famnum_samsung,
        IF(REGEXP_CONTAINS(tp_lc, r'\\biphone[[:space:]]*[0-9]{1,2}\\b'), 1, 0) AS famnum_iphone
      FROM enriched_base2
    ),

    /* **CORRECCIÓN:** Se calcula el scoring en una única etapa y se seleccionan TODAS las columnas*/
    scored AS (
      SELECT
        e.*, -- Seleccionar todas las columnas anteriores
        
        -- 1. Cálculo de Penalización de Variante
        CASE
          WHEN @want_pro=1 THEN CASE
            WHEN has_pro_only=1 THEN 0
            WHEN has_pro_max=1  THEN -120 -- Penaliza si es Pro Max y solo se busca Pro
            ELSE -60
          END
          WHEN @want_plus=1 THEN CASE
            WHEN has_plus=1 THEN 0
            WHEN has_ultra=1 THEN -120 -- Penaliza si es Ultra y solo se busca Plus
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

        -- 2. Cálculo de Penalización de Número Inconsistente (Apple)
        CASE
          WHEN @brand_lock = TRUE AND @brand_kw = 'iphone' AND @qnum IS NOT NULL AND @qnum != '' THEN
            CASE
              WHEN e.model_number IS NULL THEN 0 -- Usar alias de tabla e.
              WHEN CAST(@qnum AS INT64) = e.model_number THEN 0 -- Usar alias de tabla e.
              ELSE -80
            END
          ELSE 0
        END AS apple_num_mismatch_penalty,

        -- 3. Cálculo de Match Score Final
        (
          50 * ( -- Peso alto a los tokens de modelo específico (e.g. tok_a55, tok_ip17pro)
            e.tok_ip17promax + e.tok_ip17pro + e.tok_ip17plus + e.tok_ip17 +
            e.tok_ip16promax + e.tok_ip16pro + e.tok_ip16 + e.tok_ip15 + e.tok_ip14 +
            e.tok_s25ultra + e.tok_s25plus + e.tok_s25fe + e.tok_s25 +
            e.tok_s24ultra + e.tok_s24plus + e.tok_s24 +
            e.tok_a56 + e.tok_a55 + e.tok_a36 + e.tok_a35 + e.tok_a17 + e.tok_a16 + e.tok_a15
          )
          + 10 * (e.famnum_samsung + e.famnum_iphone) -- Peso medio a tokens genéricos (e.g. galaxy s)
          + 3  * IF(@prio79, e.plan_max_7990, 0)
          + 8  * e.cap_hit -- MEJORA: Mayor peso al almacenamiento explícito
          + 1  * ARRAY_LENGTH(ARRAY(SELECT t FROM UNNEST(@terms) AS t WHERE STRPOS(e.tp_lc, t) > 0))
          + (CASE WHEN @want_pro=1 THEN (CASE WHEN e.has_pro_only=1 THEN 0 ELSE -120 END) WHEN @want_pro_max=1 THEN (CASE WHEN e.has_pro_max=1 THEN 0 ELSE -120 END) ELSE 0 END) -- Peor caso, usar lógica inline para variant
          + (CASE WHEN @brand_lock = TRUE AND @brand_kw = 'iphone' AND @qnum IS NOT NULL AND @qnum != '' THEN (CASE WHEN e.model_number IS NULL THEN 0 WHEN CAST(@qnum AS INT64) = e.model_number THEN 0 ELSE -80 END) ELSE 0 END) -- Peor caso, usar lógica inline para mismatch
          -- Se usa el nombre de las columnas calculadas en el CTE scored
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

    SELECT * -- Selecciona todas las columnas, incluidas las de scoring
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
      -- Ordenamiento prioriza la coincidencia de tokens específicos y luego la penalización de variante
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
        # (Opcional) Relaxed query: dejamos como estaba antes, ya con brand_lock activo
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

    # Fallback cercanos (misma marca, modelo ±2)
    if (not rows or not had_exact) and (canon and qnum and qnum.isdigit()):
        q = int(qnum)
        # Ampliamos a +/- 2 generaciones
        near_nums = [str(max(1, q-2)), str(max(1, q-1)), str(q+1), str(q+2)]
        
        # Filtramos números fuera del rango S/A (10-30) para evitar sugerir modelos obsoletos
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
        ORDER BY COALESCE(price_list,9e18) ASC
        LIMIT 20
        """
        jc2 = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("brand_aliases", "STRING", brand_aliases or []),
                bigquery.ArrayQueryParameter("near_nums", "STRING", near_nums),
            ]
        )
        near_rows = list(bq_client.query(near_sql, job_config=jc2).result())
        if near_rows:
            rows = near_rows
            had_exact = False # Si entra aquí, no fue un "exact match"

    cand = []
    for r in rows:
        cand.append({
            "id": r["id"], "marca": r["brand"], 
            "producto": r["title_product"][:38],
            "imagen": r["image"],
            "url": r["calltoaction_url"], "tipo": "Celular",
            "plan": r["title_plan"], "price_list": r["price_list"], "discount": r["discount"],
            "porcentaje_descuento": None, "line": r["line"], "modality": r["modality"], "financing": r["financing"],
        })

    allowed = None if explicit_mod else ["Portabilidad","Renovación"] if not want_prepago else None
    top = _pick_by_modality_priority(cand, k=5, need_porta=2, allowed_modalities=allowed)
    return top, had_exact, nq

# ------------------ BQ: Marca-solo (TOP5 último mes) ------------------
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

        -- flags por categoría
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
    LIMIT 12
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
    
    # 1. Búsqueda estricta por marca
    if brand_focus:
        res_brand = _run_accessories_query(terms, brand_focus=brand_focus, require_brand=True, strict_brand=False, cat_hint=cat_hint)
        for item in res_brand:
            unique_items.setdefault(item['id'], item)

    # 2. Búsqueda por categoría (genérica)
    if cat_hint:
        res_generic_cat = _run_accessories_query(terms, brand_focus=None, require_brand=False, strict_brand=False, cat_hint=cat_hint)
        for item in res_generic_cat:
            unique_items.setdefault(item['id'], item)

    # 3. Relleno para términos muy genéricos (cargador)
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
    # Determinar si es postpago o prepago
    is_prepago = _has_prepago_term(nq) or 'prepago' in nq.lower()
    line_filter_product = "Claro Chip Claro Prepago" if is_prepago else "Claro Chip Claro Postpago"
    line_label = "Postpago" if not is_prepago else "Prepago"
    
    # Se buscará en title_product por "Claro Chip Claro Postpago" o "Claro Chip Claro Prepago"
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

# ------------------ Claro Video ------------------
def build_claro_video_response(normalized_query: str) -> dict:
    """
    Construye la respuesta para categoría 'pelicula' usando el catálogo de Claro video en BigQuery.

    - Usa search_claro_video_catalog() para obtener:
        * Lista de películas principales (coincidencias fuertes) -> 'listado'
        * Lista de relacionadas (otras recomendadas) -> 'relacionados'
    - Si no encuentra nada, cae en un fallback informativo estático.
    """

    try:
        main_movies, related_movies = search_claro_video_catalog(normalized_query)
    except Exception as e:
        logger.exception("Error construyendo respuesta de Claro video", exc_info=e)
        main_movies, related_movies = [], []

    # -------- Caso: no se encontró match claro en catálogo --------
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

    # -------- Caso: SÍ se encontraron películas --------

    first_movie = main_movies[0]
    titulo_pelicula = (
        first_movie.get("title")
        or first_movie.get("title_original")
        or "esta película"
    )

    titulo_marketero = f"¡Disfruta «{titulo_pelicula}» en Claro video!"

    # LISTADO: todas las coincidencias "principales"
    listado = []
    for m in main_movies:
        listado.append(
            {
                "nombre": m.get("title") or m.get("title_original"),
                "texto": m.get("description_large") or m.get("description"),
                "url": m.get("url"),
                "imagen": (
                    m.get("image_medium")
                    or m.get("image_small")
                    or m.get("image_large")
                ),
            }
        )

    # RELACIONADOS: el resto
    relacionados = []
    for m in related_movies:
        relacionados.append(
            {
                "nombre": m.get("title") or m.get("title_original"),
                "texto": m.get("description_large") or m.get("description"),
                "url": m.get("url"),
                "imagen": (
                    m.get("image_medium")
                    or m.get("image_small")
                    or m.get("image_large")
                ),
            }
        )

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

    cors_resp = origin_check(
        request,
        allowed=[
            "https://www.claro.com.pe",
            "https://test-claro-pe.prod.clarodigital.net",
            "https://search-test-1079186964678.us-central1.run.app",
        ],
    )
    if cors_resp:
        return cors_resp

    if not user_query:
        return jsonify({"error": "El parámetro 'q' es requerido"}), 400

    user_query = sanitize_query(user_query)

    try:
        normalized_query, category = classify_with_gemini(user_query)

        if category == "pelicula":
            respuesta = build_claro_video_response(normalized_query)
            respuesta["tipo_respuesta"] = "general"
            respuesta["_meta"] = {"normalized_query": normalized_query, "category": category, "cache_hit": False}
            guardar_pregunta_en_historial(normalized_query, sia_id, respuesta, pregunta_timestamp, user_agent)
            return jsonify(respuesta)

        respuesta = None
        uso_cache = False

        if category == "general":
            cache = buscar_respuesta_definitiva(normalized_query)
            if cache:
                uso_cache = True
                respuesta = json.loads(cache)
        
        # --- Planes Móviles ---
        if respuesta is None and category == "planes":
            planes = search_plans_bq(normalized_query)
            plan_type = "Postpago" if not _has_prepago_term(normalized_query) and 'postpago' in normalized_query.lower() else "Prepago"
            
            # Se usa el motor de Vertex para generar un resumen
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
            guardar_en_respuestas_definitivas(normalized_query, respuesta)
            respuesta["_meta"] = {"normalized_query": normalized_query, "category": category, "cache_hit": uso_cache}
            guardar_pregunta_en_historial(normalized_query, sia_id, respuesta, pregunta_timestamp, user_agent)
            return jsonify(respuesta)


        # --- tienda_productos (accesorios/routers) ---
        if respuesta is None and category == "tienda_productos":
            producto_tp = search_accessories_bq(normalized_query)

            # Forzar Lenovo a tienda_productos
            if brand_name == "lenovo":
                respuesta = {
                    "titulo": f"¡Ofertas Exclusivas para {normalized_query} en Tienda Claro!",
                    "descripcion": f"Descubre lo que tenemos para ti: ¡lo último en tecnología te espera!",
                    "producto": producto_tp,
                    "recomendados": DEFAULT_RECOMMENDATIONS,
                    "status": "Found" if producto_tp else "Not Found",
                    "tipo_respuesta": "tienda",
                }
            elif producto_tp:
                respuesta = {
                    "titulo": f"¡Encontramos lo que buscabas para {normalized_query}!",
                    "descripcion": "¡Potencia tu mundo con la mejor tecnología! Mira estas opciones irresistibles.",
                    "producto": producto_tp,
                    "recomendados": DEFAULT_RECOMMENDATIONS,
                    "status": "Found",
                    "tipo_respuesta": "tienda",
                }
            else:
                respuesta = {
                    "titulo": f"Ups, no encontramos resultados para {normalized_query}",
                    "descripcion": "No encontramos coincidencias exactas en accesorios o productos de tienda. Explora nuestras recomendaciones:",
                    "producto": [],
                    "recomendados": DEFAULT_RECOMMENDATIONS,
                    "status": "Not Found",
                    "tipo_respuesta": "tienda",
                }
            respuesta["_meta"] = {"normalized_query": normalized_query, "category": category, "cache_hit": uso_cache}
            guardar_pregunta_en_historial(normalized_query, sia_id, respuesta, pregunta_timestamp, user_agent)
            return jsonify(respuesta)
        
        # --- Marca sola → TOP5 ---
        brand_name = _canonical_brand_from_query(normalized_query)
        # MEJORA: Se añade 'poco' y 'redmi' al check de marca simple
        is_simple_brand_query = normalized_query in ["iphone", "samsung", "galaxy", "xiaomi", "redmi", "poco", "motorola", "moto"]
        only_brand_corrected = (
            brand_name is not None
            and len(_nums_from_query(normalized_query)) == 0
            and (is_simple_brand_query or not _match_any(PHONE_MODEL_PATTERNS, normalized_query))
        )
        
        # --- Celulares / Marca Simple Top 5 ---
        if respuesta is None:
            if category == "celulares" and only_brand_corrected:
                top5_brand_items, mes_label = search_brand_top5_products(normalized_query)
                if top5_brand_items:
                    accesorios = search_accessories_bq(normalized_query)

                    brand_cap = _canonical_brand_from_query(normalized_query).capitalize()
                    desc = f"Descubre los equipos más populares de {brand_cap}. ¡Tendencias de {mes_label}! Llévalo con un **Plan Max Ilimitado 79.90** para disfrutar al máximo."

                    respuesta = {
                        "titulo": f"¡Los Más Vendidos de {brand_cap} están en Claro!",
                        "descripcion": desc,
                        "producto": top5_brand_items,
                        "recomendados": accesorios if accesorios else DEFAULT_RECOMMENDATIONS,
                        "status": "Found",
                        "tipo_respuesta": "tienda",
                    }
                    respuesta["_meta"] = {"normalized_query": normalized_query, "category": category, "cache_hit": uso_cache}
                    guardar_pregunta_en_historial(normalized_query, sia_id, respuesta, pregunta_timestamp, user_agent)
                    return jsonify(respuesta)

            # Lógica para búsqueda de modelos específicos
            if category == "celulares":
                topN, had_exact, wanted_label = search_products_bq(
                    normalized_query,
                    prioritize_plan_79=(category == "celulares")
                )

                # Recomendados por marca (si se detecta)
                accesorios = search_accessories_bq(normalized_query)
                recomendados = accesorios if accesorios else DEFAULT_RECOMMENDATIONS
                
                if topN:
                    # Mensaje más marketero
                    if had_exact:
                        desc = f"¡Lo tenemos! El {wanted_label.capitalize()} te espera. Descubre nuestras ofertas exclusivas en Tienda Claro"
                    else:
                        desc = f"¡Te mostramos opciones irresistibles! Por el momento no contamos con el {wanted_label.capitalize()} exacto, pero estos modelos cercanos te encantarán."
                    
                    respuesta = {
                        "titulo": f"¡Encuentra tu {wanted_label.capitalize()} en Tienda Claro!",
                        "descripcion": desc,
                        "producto": topN,
                        "recomendados": recomendados,
                        "status": "Found",
                        "tipo_respuesta": "tienda",
                    }
                else:
                    respuesta = {
                        "titulo": f"Ups, no encontramos resultados para {normalized_query}",
                        "descripcion": f"Lo sentimos, no encontramos el **{normalized_query.capitalize()}** en nuestra tienda. ¡Pero no te preocupes! Tenemos alternativas increíbles:",
                        "producto": [],
                        "recomendados": recomendados,
                        "status": "Not Found",
                        "tipo_respuesta": "tienda",
                    }
            else:
                # General → Vertex
                respuesta = get_summary_from_vertex(normalized_query)
                respuesta["query"] = normalized_query
                respuesta["tipo_respuesta"] = "general"
                guardar_en_respuestas_definitivas(normalized_query, respuesta)

        guardar_pregunta_en_historial(normalized_query, sia_id, respuesta, pregunta_timestamp, user_agent)
        respuesta["_meta"] = {"normalized_query": normalized_query, "category": category, "cache_hit": uso_cache}
        return jsonify(respuesta)

    except Exception as e:
        logger.error(f"[dev] Error procesando '{user_query}': {e}", exc_info=True)
        return jsonify({
            "titulo": "Error inesperado",
            "descripcion": "<p>Ocurrió un error al procesar tu solicitud.</p>",
            "query": user_query,
            "status": "Error",
            "tipo_respuesta": "general",
            "error_detalle": str(e)
        }), 500