import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ============================================
# CONFIGURACION
# ============================================

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Falta GEMINI_API_KEY.  Crea archivo .env con tu key")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# ============================================
# CARGAR DATOS Y MODELO
# ============================================

@st.cache_resource
def cargar_modelo():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def cargar_datos():
    df = pd.read_csv("data/gold/failures_for_embeddings.csv")
    embeddings = np.load("data/gold/embeddings.npy")
    return df, embeddings

modelo = cargar_modelo()
df_gold, embeddings_existentes = cargar_datos()

# ============================================
# FUNCIONES RAG
# ============================================

def buscar_similares(descripcion_usuario, top_k=5):
    embedding_usuario = modelo.encode([descripcion_usuario])
    similitudes = cosine_similarity(embedding_usuario, embeddings_existentes)[0]
    indices_top = np.argsort(similitudes)[-top_k:][::-1]
    
    resultados = []
    for idx in indices_top:
        resultados.append({
            'nombre': df_gold. iloc[idx]['name'],
            'texto': df_gold. iloc[idx]['text'],
            'similitud': similitudes[idx]
        })
    
    return resultados

def generar_prediccion(descripcion_usuario, startups_similares):
    contexto = "STARTUPS QUE MURIERON DE FORMA SIMILAR:\n\n"
    for i, startup in enumerate(startups_similares, 1):
        contexto += f"{i}. {startup['nombre']}:\n{startup['texto']}\n\n"
    
    prompt = f"""Eres el ORACULO DE LA MUERTE DE STARTUPS. 
Tu trabajo es predecir como morira una startup basandote en startups similares que ya fracasaron.  

DESCRIPCION DE LA STARTUP DEL USUARIO:
{descripcion_usuario}

{contexto}

Genera una prediccion SARCASTICA y DIVERTIDA que incluya:

1. **PROBABILIDAD DE MUERTE**: (porcentaje del 1-100%)
2. **CAUSA DE MUERTE**: (nombre creativo y sarcastico)
3. **TIEMPO ESTIMADO**: (cuanto tardara en morir)
4.  **ULTIMAS PALABRAS**: (que diran los founders)
5. **EPITAFIO**: (que pondra en su tumba)
6. **LECCION**: (que pueden aprender)

Se creativo, sarcastico pero informativo. Basa tu prediccion en los patrones de las startups similares."""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    
    return response. text

# ============================================
# INTERFAZ STREAMLIT
# ============================================

st. set_page_config(
    page_title="Startup Death Oracle",
    page_icon="💀",
    layout="centered"
)

st. title("💀 Oraculo de la Muerte de Startups")
st.markdown("*Predice como morira tu startup basandose en 409 startups que ya fracasaron*")

st.divider()

st.subheader("Describe tu startup")
descripcion = st.text_area(
    "Que hace tu startup?",
    placeholder="Ej: App de delivery de comida que compite con Uber Eats usando IA.. .",
    height=100
)

if st.button("Consultar el Oraculo", type="primary", use_container_width=True):
    if descripcion:
        with st.spinner("El oraculo esta consultando los espiritus de startups muertas..."):
            similares = buscar_similares(descripcion, top_k=5)
            prediccion = generar_prediccion(descripcion, similares)
        
        st.divider()
        
        st.subheader("💀 La Profecia del Oraculo")
        st.markdown(prediccion)
        
        st. divider()
        
        with st.expander("Startups similares que ya murieron"):
            for startup in similares:
                st.markdown(f"**{startup['nombre']}** (Similitud: {startup['similitud']:.1%})")
                st.caption(startup['texto'][:200] + "...")
                st.divider()
    else:
        st.warning("Escribe una descripcion de tu startup primero")

st.divider()
st.caption("Creado con datos de 409 startups fallidas | RAG + Gemini")