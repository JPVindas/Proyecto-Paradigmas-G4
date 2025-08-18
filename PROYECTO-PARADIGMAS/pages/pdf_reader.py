import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import plotly.express as px
import google.generativeai as genai

import PyPDF2
import io
import json

# :: inicio CONFIGURACION GENERAL ::

genai.configure(api_key="AIzaSyB77sw3lhzRhfRrdFMntOhxRLciX9wuuxU") # configuracion API Gemini para asistente IA.
st.set_page_config(page_title="Análisis de Documentos", layout="wide")

# :: fin CONFIGURACION GENERAL :: 

# :: inicio VISTA GENERAL ANALISIS DE DOCUMENTOS ::

st.markdown(
    "<h1 style='text-align: center;'>Análisis Automatizado de Documentos</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Puedes subir cualquier documento tipo .PDF, .DOC, .DOCX, .RTF o .TXT para que Gemini haga un resúmen de lo insertado.</p>", 
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["Análisis automático", "Asistente IA (Gemini)"])

# :: inicio ESTRUCTURA TAB1 ::
with tab1:
    archivo = st.file_uploader(
        "Selecciona un archivo de tu explorador o arrástrarlo a la vista para comenzar tu análisis.", 
        type=["pdf", "txt"], # solo permite subir archivos .PDF y .TXT
        key="fileuploader"
    )
    df = None

    # Comando borra archivo anterior cuando se introduce uno nuevo.
    # Permite que la info. no se mezcle.
    st.session_state['texto_extraido'] = None

    if archivo: # el usuario inserta el archivo.
        try:
            if archivo.name.endswith(".txt"): 
                archivo.seek(0) # lee el archivo desde la primera fila. 
                try:
                    df = pd.read_csv(archivo, sep=None, engine='python')
                except Exception:
                    archivo.seek(0)
                    txt = archivo.read().decode('utf-8', errors='ignore') # de binario a texto legible y omite caracteres invalidos
                    st.markdown(
                        "<h2 style='text-align: center;'>Archivo .TXT cargado exitosamente</h2>", 
                        unsafe_allow_html=True
                    )
                    st.session_state['texto_extraido'] = txt # guarda el archivo en una variable para no tener que cargarlo otra vez.
            elif archivo.name.endswith(".pdf"):
                pdf_reader = PyPDF2.PdfReader(archivo)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                if text.strip():
                    st.session_state['texto_extraido'] = text

                df = None
            elif df is None:
                st.info("No se pudo cargar como tabla estructurada. Si es texto plano, revisa el contenido mostrado arriba.")

        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")

        texto_extraido = st.session_state.get('texto_extraido', "") # llamamos a la variable anteriormente establecida.

        # prompt predeterminado para mostrar resumen del contenido del archivo
        prompt = (
            f"Tengo el siguiente texto extraído de un archivo (puede ser PDF, TXT, etc):\n"
            f"{texto_extraido}\n"
            f"Necesito que hagas un resumen del archivo. Por favor, responde de manera breve y profesional pero divide la información en párrafos y bulletpoints si es necesario."
        )
        try:
            modelo = genai.GenerativeModel('models/gemini-1.5-flash-latest') # modelo de gemini que se esta usando.
            respuesta = modelo.generate_content(prompt)
            st.write("**Respuesta Generada por Gemini 1.5:**")
            # respuesta generada a raiz del prompt y texto extraido
            st.markdown(
                f"<div style='text-align: justify;'>{respuesta.text}</div>",
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error al comunicarse con Gemini: {e}")

    elif 'df' in st.session_state:
        df = st.session_state['df']
# :: fin ESTRUCTURA TAB1 ::

# :: inicio ESTRUCTURA TAB2 ::
with tab2:
    st.markdown(
        "<h2 style='text-align: center;'>Asistente Personal - Gemini 1.5</h2>", 
        unsafe_allow_html=True
    )

    df = st.session_state.get('df', None)
    texto_extraido = st.session_state.get('texto_extraido', None)

    if (texto_extraido is not None and len(texto_extraido.strip()) > 0):
        st.markdown(
            "<p style='text-align: center;'>Realiza preguntas a <span style='font-style: italic;'>Gemini</span> para obtener el mejor entendimiento de la información del archivo insertado.</p>",
            unsafe_allow_html=True
        )

        # cuadro de texto donde usuario ingresará la pregunta.
        pregunta = st.text_input("", placeholder="Ej: ¿Cuál es el tema principal del documento?", key="pregunta_gemini")

        if st.button("Realizar Pregunta"):
            if texto_extraido:
                # prompt predeterminado similar al de ventana de Analisis
                prompt = (
                    f"Tengo el siguiente texto extraído de un archivo (puede ser PDF, TXT, etc):\n"
                    f"{texto_extraido[:3000]}\n"
                    f"Pregunta: {pregunta}\nPor favor, responde de manera breve y profesional pero divide la información en párrafos y bulletpoints si es necesario a no ser que se te indique lo contrario."
                )
            else:
                st.warning("No hay datos ni texto cargado para enviar a la IA.")
                st.stop()

            try:
                # generacion de respuesta similar al de ventana de Analisis
                modelo = genai.GenerativeModel('models/gemini-1.5-flash-latest')
                respuesta = modelo.generate_content(prompt)
                st.markdown(
                    "<h3 style='text-align: center;'>Respuesta de Gemini 1.5</h3>", 
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div style='text-align: justify;'>{respuesta.text}</div>",
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error al comunicarse con Gemini: {e}")
    else:
        st.markdown(
            "<p style='text-align: center;'>/// Primero sube un archivo de datos o texto en la pestaña 'Análisis automático' para habilitar el asistente Gemini. ///</p>",
            unsafe_allow_html=True
        )
# :: fin ESTRUCTURA TAB2 ::

