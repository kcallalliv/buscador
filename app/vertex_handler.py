from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
import logging
import json

logger = logging.getLogger(__name__)

PROJECT_ID = "prd-claro-mktg-data-storage"
LOCATION = "global"
ENGINE_ID = "claro-peru_1748977843565"

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
        "Si no encuentras información relacionada a Claro Perú o la consulta no es relevante, "
        'devuelve igualmente el JSON completo con "listado" y "relacionados" vacíos y "status": "Not Found". '
        "No incluyas explicaciones adicionales fuera del JSON. "
        "IMPORTANTE: El campo 'texto' dentro de 'listado' debe tener máximo 170 caracteres y 'relacionados' debe tener máximo 130 caracteres."
        "IMPORTANTE: Priorizar siempre los resultados para los servicios hacia personas sobre las de negocios o empresas en listado y relacionados, ten en cuenta de no repetir las mismas urls"
        "IMPORTANTE: Cuando las consultas son sobre celulares debes tomar la iniciativa de buscar cerrar ventas y dirigir hacia https://www.tiendaclaro.pe/"
        "IMPORTANTE: En relacionados toma en cuenta que tiene que tener coherencia el nombre y la url que le proporcionas"
        "IMPORTANTE: Si la consulta está relacionada con **restaurantes, eventos, teatro o circos**, prioriza en listado y relacionados enlaces a **Claro Club**"
        "IMPORTANTE: Si la consulta está relacionada con **nombres de películas o series**, prioriza en listado y relacionados enlaces a **Claro Video**"
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

    response = client.answer_query(request)

    vertex_answer_text = response.answer.answer_text.strip()

    # Intenta limpiar bloques de código Markdown
    if vertex_answer_text.startswith("```json"):
        vertex_answer_text = vertex_answer_text[len("```json"):].strip()
        if vertex_answer_text.endswith("```"):
            vertex_answer_text = vertex_answer_text[:-len("```")].strip()
    elif vertex_answer_text.startswith("```"): # Para el caso de ``` sin "json"
        vertex_answer_text = vertex_answer_text[len("```"):].strip()
        if vertex_answer_text.endswith("```"):
            vertex_answer_text = vertex_answer_text[:-len("```")].strip()
    # Si por alguna razón el modelo solo pone "json" al principio sin comillas inversas
    elif vertex_answer_text.startswith("json"):
        vertex_answer_text = vertex_answer_text[len("json"):].strip()

    # Después de la limpieza, el texto debería ser solo el JSON
    logger.info(f"Respuesta limpia recibida del modelo: {vertex_answer_text}")

    parsed_data = {
        "titulo": "Respuesta de Claro Perú",
        "descripcion": "<p>Lo sentimos, no se pudo procesar la respuesta del modelo.</p>",
        "listado": [],
        "relacionados": [],
        "status": "Not Found"
    }

    try:
        if not vertex_answer_text.endswith("}"):
            raise ValueError("Respuesta incompleta o malformada (falta cierre de JSON)")

        model_generated_data = json.loads(vertex_answer_text)
        parsed_data.update(model_generated_data)

        # Log resumido: solo campos clave
        logger.info(
            f"Título: {parsed_data.get('titulo', 'Sin título')} | "
            f"Status: {parsed_data.get('status', 'Desconocido')} | "
            f"Listados: {len(parsed_data.get('listado', []))} | "
            f"Relacionados: {len(parsed_data.get('relacionados', []))}"
        )

    except Exception as e:
        logger.error(f"Error al parsear el JSON, devolviendo Not Found. Detalle: {e}")

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