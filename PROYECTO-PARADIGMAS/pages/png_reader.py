import streamlit as st

import google.generativeai as genai # usar API de Gemini

from PIL import Image
import io

# :: inicio CONFIGURACION GENERAL ::

genai.configure(api_key="AIzaSyDzzTT-tQLGAFlEVJJx0_Uhir-TbATgVyc") # configuracion API Gemini para asistente IA.
st.set_page_config(page_title="Análisis de Imágenes", layout="wide")

# :: fin CONFIGURACION GENERAL :: 

# :: inicio VISTA GENERAL ANALISIS DE IMAGENES ::

st.markdown(
    "<h1 style='text-align: center;'>Análisis Automatizado de Imágenes</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Puedes subir cualquier imagen tipo .PNG, .JPG o .JPEG para que Gemini interprete lo insertado.</p>", 
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["Análisis automático", "Asistente IA (Gemini)"])

# :: inicio ESTRUCTURA TAB1 ::
with tab1:
    archivo = st.file_uploader(
        "Selecciona un archivo de tu explorador o arrástrarlo a la vista para comenzar tu análisis.", 
        type=["png", "jpg", "jpeg"], # solo permite subir archivos .PNG, .JPG o .JPEG
        key="fileuploader"
    )

    # Comando borra archivo anterior cuando se introduce uno nuevo.
    # Permite que la info. no se mezcle.
    st.session_state['texto_extraido'] = None

    if archivo: # el usuario inserta el archivo.
        try:
            img = Image.open(archivo)

            # convierte imagen a bytes (Gemini solo puede leer en bytes).
            img_bytes = io.BytesIO()
            #independientemente del formato, la iamgen pasa a ser .PNG para una lectura más simple.
            img.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()

            # guarda la imagen (en bytes) en una variable para no tener que cargarla otra vez.
            st.session_state['imagen_extraida'] = img_bytes

            st.markdown(
                "<h2 style='text-align: center;'>Imagen procesada exitosamente</h2>", 
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")

        if 'imagen_extraida' in st.session_state:
            img_bytes = st.session_state['imagen_extraida']
            try:
                modelo = genai.GenerativeModel('models/gemini-1.5-flash-latest') # modelo de gemini que se esta usando.

                # prompt predeterminado para mostrar resumen del contenido de la imagen
                prompt = "Neceisto que analices esta imagen y resumas detalladamente lo que contiene. Por favor, responde de manera breve y profesional pero divide la información en párrafos y bulletpoints si es necesario."
                
                # metodo para que Gemini pueda interpretar texto + imagen con libreria google-generativeai.
                # mime_type es el tipo de archivo y en el data se pasan los bytes de la imagen.
                respuesta = modelo.generate_content([
                    prompt,
                    {"mime_type": "image/png", "data": img_bytes}
                ])
                
                st.markdown(
                    f"<div style='text-align: justify;'>{respuesta.text}</div>",
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error al comunicarse con Gemini: {e}")

# :: fin ESTRUCTURA TAB1 ::

# :: inicio ESTRUCTURA TAB2 ::
with tab2:
    st.markdown(
        "<h2 style='text-align: center;'>Asistente Personal - Gemini 1.5</h2>", 
        unsafe_allow_html=True
    )

    if "imagen_extraida" in st.session_state:
        st.markdown(
            "<p style='text-align: center;'>Realiza preguntas a <span style='font-style: italic;'>Gemini</span> para obtener el mejor entendimiento de la imagen insertada.</p>",
            unsafe_allow_html=True
        )

        # cuadro de texto donde usuario ingresará la pregunta.
        pregunta = st.text_input("", placeholder="Ej: ¿Qué objetos aparecen en la imagen?", key="pregunta_gemini")

        if st.button("Realizar Pregunta"):
            if st.session_state["imagen_extraida"]:
                img_bytes = st.session_state["imagen_extraida"]

                # prompt predeterminado similar al de ventana de Analisis
                prompt = (
                    f"Tengo esta imagen subida por el usuario.\n"
                    f"El ususario pregunta: {pregunta}\n"
                    f"Por favor, responde de manera breve y profesional"
                    f"pero divide la información en párrafos y bulletpoints"
                    f"si es necesario a no ser que se te indique lo contrario."
                    f"Recuerda utilizar un análisis lógico y crítico, no asumas"
                    f"nada de lo que no estás seguro."
                )

                try:
                    # generacion de respuesta similar al de ventana de Analisis
                    modelo = genai.GenerativeModel('models/gemini-1.5-flash-latest')

                    # metodo para que Gemini pueda interpretar texto + imagen con libreria google-generativeai.
                    # mime_type es el tipo de archivo y en el data se pasan los bytes de la imagen.
                    respuesta = modelo.generate_content([
                        prompt,
                        {"mime_type": "image/png", "data": img_bytes}
                    ])

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
            "<p style='text-align: center;'>/// Primero sube una imagen en la pestaña 'Análisis automático' para habilitar el asistente Gemini. ///</p>",
            unsafe_allow_html=True
        )
# :: fin ESTRUCTURA TAB2 ::

