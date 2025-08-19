# leer_csv.py
import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import os
import tempfile

# =========================
# Importar funciones de app.py
# =========================
from app import (
    cargar_dataframe,
    detectar_tipos,
    generar_histograma_claro,
    generar_boxplot_claro,
    generar_correlacion_clara,
    generar_barras_claras,
    generar_violin_y_box,
    detectar_outliers,
    clustering_kmeans,
    plot_clusters_pca,
)

# =========================
# Configuración de la página
# =========================
st.set_page_config(
    page_title="🔎 Análisis Automatizado de Datos",
    layout="wide"  # Esto hace que toda la app use el ancho completo
)

# Configurar API Gemini
genai.configure(api_key="AIzaSyDzzTT-tQLGAFlEVJJx0_Uhir-TbATgVyc")

# =========================
# Título
# =========================
st.title("🔎 Análisis Automatizado de Datos")
st.markdown("Sube tu archivo CSV, Excel, PDF, TXT o JSON y descubre insights al instante.")

# =========================
# Función para seleccionar variables representativas
# =========================
def seleccionar_variables(df, num_cols, cat_cols):
    if num_cols:
        var_varianza = df[num_cols].var().sort_values(ascending=False)
        selected_nums = var_varianza.index[:min(3, len(var_varianza))].tolist()
    else:
        selected_nums = []

    selected_cat = None
    best_nonnull = -1
    for c in cat_cols:
        non_null = df[c].notna().sum()
        unique_ct = df[c].nunique(dropna=True)
        if unique_ct <= 50 and non_null > best_nonnull:
            selected_cat = c
            best_nonnull = non_null

    return selected_nums, selected_cat

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["Análisis automático", "🤖 Asistente IA"])

# =========================
# Tab 1: Análisis automático
# =========================
with tab1:
    archivo = st.file_uploader(
        "Sube CSV, Excel, JSON, PDF o TXT",
        type=["csv", "xlsx", "json", "pdf", "txt"]
    )
    df = None
    texto_extraido = ""

    if archivo:
        name = archivo.name.lower()
        try:
            if name.endswith((".csv", ".xlsx", ".json")):
                with st.spinner("Cargando datos..."):
                    df = cargar_dataframe(archivo)
                    st.session_state['df'] = df
                    st.success(f"Datos cargados: {len(df)} filas × {len(df.columns)} columnas")
                    st.dataframe(df.head(min(10, len(df))), use_container_width=True)

            elif name.endswith(".pdf"):
                with st.spinner("Extrayendo texto de PDF..."):
                    texto_extraido = extraer_texto_pdf(archivo)
                    st.session_state['texto_extraido'] = texto_extraido
                    st.text_area("Texto extraído (preview)", value=texto_extraido[:3000], height=200)

            else:  # TXT
                contenido = archivo.read().decode('utf-8', errors='ignore')
                st.session_state['texto_extraido'] = contenido
                st.text_area("Contenido (preview)", value=contenido[:3000], height=200)

        except Exception as e:
            st.error(f"Error al cargar archivo: {e}")

    # Recuperar del session_state si ya se cargó antes
    df = st.session_state.get('df', None)
    texto_extraido = st.session_state.get('texto_extraido', "")

    if df is not None and not df.empty:
        try:
            with st.spinner("Analizando estructura de datos..."):
                tipo_df = detectar_tipos(df)

            st.subheader("📊 Tipos de datos detectados")
            st.dataframe(tipo_df, use_container_width=True)

            # Columnas numéricas y categóricas
            num_cols = tipo_df[tipo_df["Tipo"] == "Numérica"]["Variable"].tolist()
            cat_cols = tipo_df[tipo_df["Tipo"] == "Categórica"]["Variable"].tolist()

            st.subheader("📈 Estadísticas descriptivas (numéricas)")
            st.dataframe(df[num_cols].describe(), use_container_width=True)

            st.subheader("🔍 Visualizaciones automáticas")
            selected_nums, selected_cat = seleccionar_variables(df, num_cols, cat_cols)

            # Mostrar variables seleccionadas
            st.markdown("**Variables usadas en gráficas:**")
            cols_info_left, cols_info_right = st.columns([1, 1])  # Ocupando todo el ancho
            with cols_info_left:
                st.write("Numéricas (por varianza):")
                if selected_nums:
                    for v in selected_nums:
                        st.write(f"- {v} (var={df[v].var():.4g})")
                else:
                    st.write("- (no hay variables numéricas)")

            with cols_info_right:
                st.write("Categórica seleccionada:")
                st.write(f"- {selected_cat}" if selected_cat else "- (no se encontró categórica)")

            # --------------------
            # Generar gráficos
            # --------------------
            for var in selected_nums:
                fig_h = generar_histograma_claro(df, var)
                if fig_h:
                    st.plotly_chart(fig_h, use_container_width=True)

            fig_box = generar_boxplot_claro(df, selected_nums, max_display=6)
            if fig_box:
                st.plotly_chart(fig_box, use_container_width=True)

            fig_corr = generar_correlacion_clara(df, num_cols)
            if fig_corr:
                st.plotly_chart(fig_corr, use_container_width=True)

            if selected_cat:
                fig_bar = generar_barras_claras(df, selected_cat, top_n=20)
                if fig_bar:
                    st.plotly_chart(fig_bar, use_container_width=True)

            if selected_nums:
                fig_vb = generar_violin_y_box(df, selected_nums[0])
                if fig_vb:
                    st.plotly_chart(fig_vb, use_container_width=True)

            # --------------------
            # Outliers
            # --------------------
            st.subheader("⚠️ Detección de outliers")
            if selected_nums:
                contamination = st.slider("Sensibilidad outliers", 0.01, 0.2, 0.05, 0.01, key="contamination_csv")
                outliers = detectar_outliers(df, selected_nums, contamination)
                if not outliers.empty:
                    frac = len(outliers) / max(1, len(df))
                    st.info(f"Outliers detectados: {len(outliers)} — {frac:.2%} del dataset")
                    st.dataframe(outliers.head(200), use_container_width=True)
                else:
                    st.success("No se detectaron outliers.")

            # --------------------
            # Clustering
            # --------------------
            st.subheader("🧩 Clustering (KMeans / MiniBatch)")
            if num_cols and len(num_cols) >= 2:
                n_clusters = st.slider("Número de clusters", 2, 12, 3, key="nclusters_csv")
                df_clusters_vis, model_info = clustering_kmeans(df, num_cols, n_clusters=n_clusters, minibatch=True)
                if df_clusters_vis is not None:
                    st.info(f"Muestra de {len(df_clusters_vis)} observaciones para visualizar clusters")
                    st.dataframe(df_clusters_vis.head(100), use_container_width=True)
                    fig_clusters = plot_clusters_pca(df_clusters_vis, model_info[2])
                    if fig_clusters:
                        st.plotly_chart(fig_clusters, use_container_width=True)
                else:
                    st.warning("No fue posible realizar clustering.")
            else:
                st.info("Se requieren al menos 2 columnas numéricas para clustering.")

        except Exception as e:
            st.error(f"Error al procesar CSV: {e}")

    else:
        st.info("Sube un archivo de datos para comenzar el análisis.")

# =========================
# Tab 2: Asistente IA Gemini
# =========================
with tab2:
    st.markdown("## 🤖 Asistente Inteligente (Gemini)")

    df = st.session_state.get('df', None)
    texto_extraido = st.session_state.get('texto_extraido', None)

    if (df is not None and not df.empty) or (texto_extraido is not None and len(texto_extraido.strip()) > 0):
        st.write("Hazle preguntas a la IA sobre tu archivo. Ejemplos:")
        st.markdown("- Para datos tabulares: '¿Qué variables parecen estar más relacionadas?'\n"
                    "- Para texto/pdf: '¿Cuáles son los temas principales?' o 'Hazme un resumen.'")

        pregunta = st.text_area("Pregunta para Gemini:", placeholder="¿Qué observas en los datos o en el texto?", key="pregunta_gemini")

        if st.button("Preguntar a Gemini"):
            if df is not None and not df.empty:
                muestra = df.head(10).to_csv(index=False)
                prompt = (
                    f"Tengo este dataset en formato CSV:\n{muestra}\n"
                    f"Pregunta: {pregunta}\nPor favor, responde como si fueras un analista de datos profesional."
                )
            elif texto_extraido:
                prompt = (
                    f"Tengo el siguiente texto extraído de un archivo:\n"
                    f"{texto_extraido[:3000]}\n"
                    f"Pregunta: {pregunta}\nPor favor, responde de manera concisa y profesional."
                )
            else:
                st.warning("No hay datos ni texto cargado para enviar a la IA.")
                st.stop()

            try:
                modelo = genai.GenerativeModel('gemini-1.5-flash')
                respuesta = modelo.generate_content(prompt)
                st.write("**Respuesta de Gemini:**")
                st.success(respuesta.text)
            except Exception as e:
                st.error(f"Error al comunicarse con Gemini: {e}")
    else:
        st.info("Primero sube un archivo de datos o texto en la pestaña 'Análisis automático' para habilitar el asistente Gemini.")
