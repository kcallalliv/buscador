from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
import logging
import json
# === IMPORTS AÑADIDOS PARA MANEJO DE ERRORES/TIMEOUT ===
from google.api_core import retry
from google.api_core import exceptions
import time 

logger = logging.getLogger(__name__)

PROJECT_ID = "prd-claro-mktg-data-storage"
LOCATION = "global"
ENGINE_ID = "claro-peru_1748977843565"

# ----------------- POLÍTICA DE REINTENTOS -----------------
# Define la política de reintentos para errores transitorios (503/500).
VERTEX_RETRY_POLICY = retry.Retry(
    predicate=retry.if_exception_type(
        exceptions.ServiceUnavailable,
        exceptions.InternalServerError,
        exceptions.DeadlineExceeded,
    ),
    initial=1.0,  
    multiplier=2.0, 
    maximum=10.0, 
    deadline=60.0 # Tiempo total máximo para todos los intentos
)


def get_summary_from_vertex(user_query):
    client_options = (
        ClientOptions(api_endpoint=f"{LOCATION}-discoveryengine.googleapis.com")
        if LOCATION != "global"
        else None
    )

    client = discoveryengine.ConversationalSearchServiceClient(client_options=client_options)

    serving_config = f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/default_collection/engines/{ENGINE_ID}/servingConfigs/default_serving_config"

    # PROMPT ROBUSTO Y EXPLÍCITO
    prompt_text = (
        "Eres un asistente amigable y útil de Claro Perú. Responde exclusivamente en formato JSON ESTRICTO "
        "con la siguiente estructura: { "
        '"titulo": string, '
        '"descripcion": string, '
        '"listado": [ { "nombre": string, "texto": string, "url": string }, ... ], '
        '"relacionados": [ { "nombre": string, "texto": string, "url": string }, ... ], '
        '"status": "Found" o "Not Found" }. '
        #"Si no encuentras información relacionada a Claro Perú o la consulta no es relevante, "
        #'devuelve igualmente el JSON completo con "listado" y "relacionados" vacíos y "status": "Not Found". '

        #"REGLA CRÍTICA PARA CONSULTAS INENTENDIBLES O SIN RESULTADOS: "
        #"Si el usuario escribe algo sin sentido (ej: 'asffsgsa'), preguntas fuera de contexto o si no encuentras "
        #"información específica en los documentos, debes responder con el siguiente enfoque: "
        #"1. Título: '¿Cómo puedo ayudarte mejor?' o 'Necesito un poco más de contexto'. "
        #"2. Descripción: Explica amablemente que no pudiste entender la consulta actual. Invítalo a reformular "
        #"indicándole que puedes ayudarle con: Información sobre servicios de Claro (Hogar, Móvil), "
        #"Dudas sobre facturación, Catálogo de celulares y accesorios, o búsqueda de películas en Claro Video. "
        #"3. Listado y Relacionados: Proporciona enlaces genéricos útiles (Tienda, Mi Claro, Ayuda). "
        #"4. Status: 'Found'. "

        "No incluyas explicaciones adicionales fuera del JSON. "
        "IMPORTANTE: El campo 'texto' dentro de 'listado' debe tener máximo 170 caracteres y 'relacionados' debe tener máximo 130 caracteres."
        "IMPORTANTE: Priorizar siempre los resultados para los servicios hacia personas sobre las de negocios o empresas en listado y relacionados, ten en cuenta de no repetir las mismas urls"
        "IMPORTANTE: Recuerda que el usuario puede querer tambien realizar streamin o transmisiones en vivo y por eso requiero servicios de internet, debemos tener claro la diferencia de los servicios de plataformas de streaming y si el cliente requiere servicios de internet para hacer streaming"
        "IMPORTANTE: Cuando las consultas son sobre celulares debes tomar la iniciativa de buscar cerrar ventas y dirigir hacia https://www.tiendaclaro.pe/"
        "IMPORTANTE: En relacionados toma en cuenta que tiene que tener coherencia el nombre y la url que le proporcionas"
        "IMPORTANTE: Si la consulta está relacionada con **restaurantes, eventos, teatro o circos**, prioriza en listado y relacionados enlaces a **Claro Club**"
        "IMPORTANTE: Si la consulta está relacionada con **nombres de películas o series o frases que consideres que pueden ser nombres de peliculas como por ejemplo(robot salvaje, destino final, dragon ball) o consultas sobre animales o sobre batallas**, prioriza en listado y relacionados enlaces a **Claro Video** y de status: Not Found"
        "IMPORTANTE: Si la consulta está relacionada con **canciones o música**, prioriza en listado y relacionados enlaces a **Claro Música**"
        "IMPORTANTE: Cuando te saluden o te den las gracias tambien busca brindar una respuesta bien formulada con el formato JSON"
        "IMPORTANTE: Si la consulta esta relacionada a (Paramount) debes mencionar que Claro video cuenta con otros servicios similares pero ya no debes mencionar Paramount en ninguna respuesta por que ya no se va a brindar ese servicio"

    )

    answer_generation_spec = discoveryengine.AnswerQueryRequest.AnswerGenerationSpec(
        ignore_adversarial_query=True,
        ignore_non_answer_seeking_query=True,
        ignore_low_relevant_content=False,
        model_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec.ModelSpec(
            model_version="stable"
        ),
        prompt_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec.PromptSpec(
            preamble=prompt_text
        ),
        include_citations=True,
        answer_language_code="es",
    )

    request = discoveryengine.AnswerQueryRequest(
        serving_config=serving_config,
        query=discoveryengine.Query(text=user_query),
        session=None,
        answer_generation_spec=answer_generation_spec,
    )

    # ----------------- LLAMADA ROBUSTA -----------------
    try:
        start_time = time.time()
        # === APLICACIÓN DE RETRY Y TIMEOUT ===
        response = client.answer_query(
            request=request,
            retry=VERTEX_RETRY_POLICY,
            timeout=45.0
        )
        end_time = time.time()
        logger.info(f"Vertex AI response time: {end_time - start_time:.2f}s")
        
    except exceptions.ServiceUnavailable as e:
        logger.error(f"FALLO DEFINITIVO: Vertex AI no respondió después de reintentos: {e}")
        return {
            "titulo": "Error de servicio de IA",
            "descripcion": "<p>Lo sentimos, el servicio de inteligencia artificial no está disponible temporalmente. Inténtalo de nuevo más tarde.</p>",
            "listado": [],
            "relacionados": [],
            "status": "Error"
        }
    except Exception as e:
        logger.error(f"Error inesperado en Vertex AI: {e}")
        raise 


    vertex_answer_text = response.answer.answer_text.strip()

    # Intenta limpiar bloques de código Markdown
    if vertex_answer_text.startswith("```json"):
        vertex_answer_text = vertex_answer_text[len("```json"):].strip()
        if vertex_answer_text.endswith("```"):
            vertex_answer_text = vertex_answer_text[:-len("```")].strip()
    elif vertex_answer_text.startswith("```"): 
        vertex_answer_text = vertex_answer_text[len("```"):].strip()
        if vertex_answer_text.endswith("```"):
            vertex_answer_text = vertex_answer_text[:-len("```")].strip()
    elif vertex_answer_text.startswith("json"):
        vertex_answer_text = vertex_answer_text[len("json"):].strip()

    # Después de la limpieza, el texto debería ser solo el JSON
    logger.info(f"Respuesta limpia recibida del modelo: {vertex_answer_text}")

    parsed_data = {
        "titulo": "¡Hola! Soy el Buscador IA de Claro y estoy aquí para ayudarte. ✨",
        "descripcion": (
            "No logré entender del todo tu consulta, pero no te preocupes, ¡vamos a encontrar lo que necesitas! "
            "¿podrías darme un poquito más de detalle? Puedo ayudarte a conocer más sobre nuestros productos y servicios:"
        ),
        "listado": [
            { 
                "nombre": "Planes Postpago", 
                "texto": "Cámbiate a Claro con los mejores beneficios, redes sociales ilimitadas y muchos gigas.", 
                "url": "https://www.claro.com.pe/personas/movil/postpago/" 
            },
            { 
                "nombre": "Planes Prepago", 
                "texto": "Disfruta de la libertad de nuestras recargas con beneficios Prepago Chévere.", 
                "url": "https://www.claro.com.pe/personas/movil/prepago/" 
            },
            {
                "nombre": "Equipos Celulares", 
                "texto": "Disfruta de la libertad de nuestras recargas con beneficios Prepago Chévere.", 
                "url": "https://www.tiendaclaro.pe/" 
            },
            {
                "nombre": "Servicios para el hogar", 
                "texto": "Disfruta de la libertad de nuestras recargas con beneficios Prepago Chévere.", 
                "url": "https://www.claro.com.pe/personas/hogar/internet/" 
            },
            {
                "nombre": "Claro Video", 
                "texto": "Disfruta de la libertad de nuestras recargas con beneficios Prepago Chévere.", 
                "url": "https://www.claro.com.pe/personas/app/claro-video/" 
            }
        ],
        "relacionados": [
            { 
                "nombre": "Tienda Claro", 
                "texto": "Renueva tu equipo con las mejores ofertas y tecnología de punta.", 
                "url": "https://www.tiendaclaro.pe/" 
            },
            { 
                "nombre": "Centro de Ayuda", 
                "texto": "Encuentra soluciones rápidas sobre tus servicios y pagos aquí.", 
                "url": "https://www.claro.com.pe/personas/atencion-al-cliente/" 
            }
        ],
        "status": "Found"  # <--- Mantenerlo en Found es vital
    }

    try:
        # Si Vertex devolvió texto, intentamos parsearlo
        if vertex_answer_text and vertex_answer_text.endswith("}"):
            model_generated_data = json.loads(vertex_answer_text)
            parsed_data.update(model_generated_data)
        else:
            # Si no hay respuesta del modelo, lanzamos error para ir al except
            raise ValueError("Respuesta de Vertex vacía")
            
    except Exception as e:
        logger.error(f"Usando respuesta amigable por defecto. Detalle: {e}")
        # Al no hacer update, se queda con el parsed_data amigable que definimos arriba

    # Procesamiento de citas estructuradas de Discovery Engine
    citations_from_discovery = []
    if response.answer.citations:
        for c in response.answer.citations:
            citation_url = None
            citation_title = ""
            citation_snippet = ""

            link = getattr(c, 'unstructured_document_link', None) or getattr(c, 'struct_document_link', None)
            if link:
                citation_url = getattr(link, 'uri', None)
                citation_title = getattr(link, 'title', "")
                citation_snippet = getattr(c, 'snippet', "")

            if citation_url:
                formatted_text = (
                    f"<p>Descubre más en <a href=\"{citation_url}\" target=\"_blank\" rel=\"noopener noreferrer\" "
                    f"class=\"text-claro-red font-semibold no-underline\">{citation_title or 'Enlace'}</a>: {citation_snippet}</p>"
                )
                citations_from_discovery.append({
                    "nombre": citation_title or "Fuente Adicional",
                    "texto": formatted_text,
                    "url": citation_url
                })

    if isinstance(parsed_data.get('listado'), list):
        parsed_data['listado'].extend(citations_from_discovery)
    else:
        parsed_data['listado'] = citations_from_discovery

    return parsed_data