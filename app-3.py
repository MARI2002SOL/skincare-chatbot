
import streamlit as st
import pymongo

from google import genai
from google.genai import types

# =======================
# PAGE CONFIG
# =======================

st.set_page_config(
    page_title="SkinCare Guru ✨",
    page_icon="💖",
    layout="centered"
)

# =======================
# CUSTOM CSS
# =======================

st.markdown("""
<style>

.main {
    background-color: #fff7fb;
}

h1 {
    color: #ff4fa3;
    text-align: center;
    font-weight: 800;
}

.stChatInputContainer {
    border-radius: 20px;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# =======================
# CONFIGURACIÓN
# =======================

GOOGLE_API_KEY = st.secrets["app"]["GOOGLE_API_KEY"]
MONGODB_URI = st.secrets["app"]["MONGODB_URI"]

if not GOOGLE_API_KEY or not MONGODB_URI:

    st.error(
        "❌ Missing GOOGLE_API_KEY or MONGODB_URI"
    )

    st.stop()

# =======================
# CLIENTES
# =======================

@st.cache_resource
def get_genai_client():

    return genai.Client(
        api_key=GOOGLE_API_KEY
    )

@st.cache_resource
def get_mongo_collection():

    client = pymongo.MongoClient(
        MONGODB_URI
    )

    db = client["skincare_ai_db"]

    return db["skincare_embeddings"]

client_genai = get_genai_client()

collection = get_mongo_collection()

# =======================
# EMBEDDINGS
# =======================

def crear_embedding(texto: str):

    response = client_genai.models.embed_content(
        model="gemini-embedding-001",
        contents=texto,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
        ),
    )

    return response.embeddings[0].values

# =======================
# VECTOR SEARCH
# =======================

def buscar_similares(
    embedding,
    k=5
):

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 100,
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "texto": 1,
                "categoria": 1,
                "fuente": 1,
                "score": {
                    "$meta": "vectorSearchScore"
                },
            }
        },
    ]

    return list(
        collection.aggregate(pipeline)
    )

# =======================
# GENERAR RESPUESTA
# =======================

def generar_respuesta(
    pregunta: str,
    contextos: list[dict]
) -> str:

    contexto = "\n\n".join(
        [c["texto"] for c in contextos]
    )

    prompt = f"""
You are SkinCare Guru ✨

Role:
A friendly beauty and skincare expert
who helps users understand:
- skincare ingredients
- routines
- acne treatments
- hydration
- sunscreen
- retinol
- sensitive skin care

Use ONLY the provided context.

If the answer is not found:
- clearly say so
- do not invent information

Your tone should be:
- cute
- friendly
- modern
- helpful
- concise

CONTEXT:
{contexto}

QUESTION:
{pregunta}
"""

    response = client_genai.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text

# =======================
# HEADER
# =======================

st.markdown("""
<h1>💖 SkinCare Guru ✨</h1>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;
font-size:18px;
color:#666;
margin-bottom:25px;'>

Your cute AI skincare bestie 💅<br>
Ask about routines, ingredients, acne,
hydration, retinol and more ✨

</div>
""", unsafe_allow_html=True)

# =======================
# HISTORIAL
# =======================

if "historial" not in st.session_state:

    st.session_state.historial = []

# Mostrar historial
for msg in st.session_state.historial:

    if msg["rol"] == "usuario":

        st.chat_message("user").write(
            msg["texto"]
        )

    else:

        st.chat_message("assistant").write(
            msg["texto"]
        )

# =======================
# INPUT
# =======================

pregunta = st.chat_input(
    "Ask your skincare question here... ✨"
)

# =======================
# CHAT
# =======================

if pregunta:

    # Mostrar usuario
    st.chat_message("user").write(
        pregunta
    )

    st.session_state.historial.append({
        "rol": "usuario",
        "texto": pregunta
    })

    with st.chat_message("assistant"):

        with st.spinner(
            "Finding the best skincare advice ✨"
        ):

            try:

                # Embedding
                emb = crear_embedding(
                    pregunta
                )

                # Search
                similares = buscar_similares(
                    emb,
                    k=5
                )

                # No resultados
                if not similares:

                    respuesta = (
                        "I couldn't find relevant "
                        "skincare information 💔"
                    )

                else:

                    # Gemini response
                    respuesta = generar_respuesta(
                        pregunta,
                        similares
                    )

            except Exception as e:

                respuesta = (
                    f"⚠️ Error: {e}"
                )

        st.write(respuesta)

        # =======================
        # SOURCES
        # =======================

        if 'similares' in locals() and similares:

            with st.expander(
                "🔍 View retrieved skincare context"
            ):

                for i, c in enumerate(
                    similares,
                    1
                ):

                    st.markdown(
                        f"### Fragment {i}"
                    )

                    st.markdown(
                        f"**Category:** "
                        f"{c.get('categoria', 'N/A')}"
                    )

                    st.markdown(
                        f"**Relevance Score:** "
                        f"`{c['score']:.4f}`"
                    )

                    st.write(
                        c["texto"][:500]
                        + (
                            "..."
                            if len(c["texto"]) > 500
                            else ""
                        )
                    )

                    st.divider()

    # Guardar respuesta
    st.session_state.historial.append({
        "rol": "bot",
        "texto": respuesta
    })
