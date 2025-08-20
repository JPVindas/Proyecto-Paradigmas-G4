# leer_csv.py
import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import google.generativeai as genai # type: ignore
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

# :: inicio CONFIGURACION GENERAL ::

genai.configure(api_key="AIzaSyDzzTT-tQLGAFlEVJJx0_Uhir-TbATgVyc") # configuracion API Gemini para asistente IA.
st.set_page_config(page_title="Análisis Automatizado de Datos", layout="wide")

# :: fin CONFIGURACION GENERAL :: 

# :: inicio VISTA GENERAL ANALISIS DE IMAGENES ::

st.markdown(
    "<h1 style='text-align: center;'>Análisis Automatizado de Datos</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Puedes subir cualquier archivo tipo .CSV o .XLSX para que Gemini interprete lo insertado.</p>", 
    unsafe_allow_html=True
)

# :: inicio FUNC SELECT VARIABLES REPRESENTATIVAS ::

def seleccionar_variables(df, num_cols, cat_cols):
    if num_cols:
        # varianza de las columnas numericas ordenadas de mayor a menor.
        var_varianza = df[num_cols].var().sort_values(ascending=False)
        # utiliza columnas con mayor varianza (max. 3 columnas).
        selected_nums = var_varianza.index[:min(3, len(var_varianza))].tolist()
    else:
        # lista vacia si no hay columnas numericas.
        selected_nums = []

    selected_cat = None # columna categorica resaltante
    best_nonnull = -1 
    # "por cada columna categorica"
    for c in cat_cols:
        non_null = df[c].notna().sum() # CANTIDAD de valores no nulos
        unique_ct = df[c].nunique(dropna=True) # CANTIDAD de categorias unicas (ej: soltero, casado, divorciado).

        # criterio que define la mejor columna categorica
        if unique_ct <= 50 and non_null > best_nonnull:
            selected_cat = c
            best_nonnull = non_null

    # regresa las 3 columnas numericas y la categorica seleccionada.
    return selected_nums, selected_cat

# :: fin FUNC SELECT VARIABLES REPRESENTATIVAS ::

# :: inicio FUNC SELECT VARIABLE OBJETIVO ::
def seleccionar_objetivo(df: pd.DataFrame):
    # selecciona ultima columna (variable objetivo).
    target_col = df.columns[-1]
    
    # si no es categorica, no filtra.
    if not pd.api.types.is_categorical_dtype(df[target_col]) and not df[target_col].dtype == "object":
        st.warning(f"La última columna ('{target_col}') no parece ser categórica. No se aplicará filtro.")
        return df, None
    
    # seleccion de categorias unicas para el usuario.
    st.subheader(f"Variable Objetivo: {target_col}")
    categorias = df[target_col].dropna().unique().tolist() # almacena categoias en variable
    selected_cat = st.selectbox("Selecciona la categoría con la que deseas trabajar:", categorias) # selectbox con las categorias.
    
    # target_col = categoria seleccionada por el usuario.
    # actualiza el target_col de todas las funciones.
    df_filtrado = df[df[target_col] == selected_cat]
    
    st.info(f"Trabajando con categoría **{selected_cat}** ({len(df_filtrado)} filas)")
    
    return df_filtrado, selected_cat

# :: inicio FUNC SELECT VARIABLE OBJETIVO ::

tab1, tab2 = st.tabs(["Análisis automático", "Asistente IA (Gemini)"])

# :: inicio ESTRUCTURA TAB1 ::
with tab1:
    # [archivo subido].
    archivo = st.file_uploader(
        "Selecciona un archivo de tu explorador o arrástrarlo a la vista para comenzar tu análisis.",
        type=["csv", "xls", "xlsx"] # solo puede seleccionar archivos tipo .CSV, .XLS o .XLSX
    )
    df = None
    texto_extraido = ""

    if archivo:
        name = archivo.name.lower()
        try:
            if name.endswith((".csv", ".xls", ".xlsx")):
                with st.spinner("Cargando datos..."):
                    # valor retornado del DataFrame con el [archivo subido].
                    df = cargar_dataframe(archivo)

                    # guarda el DataFrame en una variable para no tener que cargarlo otra vez.
                    st.session_state['df'] = df

                    # :: inicio MUESTRA BASICA DE DATOS ::

                    # especificaciones generales del dataframe y muestra las primeras 10 inserciones en la tabla.
                    st.success(f"Datos cargados: {len(df)} filas × {len(df.columns)} columnas")
                    st.dataframe(df.head(min(10, len(df))), use_container_width=True)

                    # :: fin MUESTRA BASICA DE DATOS ::

                    # recupera el DataFrame
                    df = st.session_state.get('df', None)

                    df, categoria_objetivo = seleccionar_objetivo(df)

                    if df is not None and not df.empty:
                        try:
                            with st.spinner("Analizando estructura de datos..."):
                                # variable que almacena el DataFrame retornado con los tipos de datos del DataFrame original.
                                tipo_df = detectar_tipos(df)

                            # :: inicio TIPOS DE DATOS DETECTADOS ::

                            st.subheader("Tipos de Datos Detectados")
                            # muestra dataframe cargado de la variable [tipo_df]
                            st.dataframe(tipo_df, use_container_width=True)

                            # almacena en variables las columnas que sean categoricas y las numericas.
                            num_cols = tipo_df[tipo_df["Tipo"] == "Numérica"]["Variable"].tolist()
                            cat_cols = tipo_df[tipo_df["Tipo"] == "Categórica"]["Variable"].tolist()

                            # :: fin TIPOS DE DATOS DETECTADOS ::

                            # :: inicio ESTADISTICAS DESCRIPTIVAS BASICAS ::

                            # variable que almacena estadisticas descriptivas basicas.
                            desc = df[num_cols].describe()

                            # cambiar el nombre predeterminado del indice.
                            desc = desc.rename(index= {
                                "count": "cant",
                                "mean": "prom",
                                "std": "std",
                                "min": "min",
                                "25%": "25%",
                                "50%": "med",
                                "75%": "75%",
                                "max": "max"
                            })

                            # genera DataFrame con estadisticas descriptivas basicas para las columnas numericas del DataFrame original.
                            st.subheader("Estadísticas Descriptivas (Numéricas)")
                            st.dataframe(desc, use_container_width=True)

                            # :: fin ESTADISTICAS DESCRIPTIVAS BASICAS ::

                            # :: inicio VISUALIZACIONES GRAFICAS ::

                            st.subheader("Visualizaciones Gráficas")
                            selected_nums, selected_cat = seleccionar_variables(df, num_cols, cat_cols)

                            # mostrar variables seleccionadas en una tabla intuitiva.
                            st.markdown("Variables Seleccionadas")
                            num_cols_display = 3
                            data = {f"Numérica {i+1}": [] for i in range(num_cols_display)}
                            data["Categórica"] = []

                            for i in range(0, len(selected_nums), num_cols_display):
                                fila = []
                                # variables numericas divididas en 3 columnas.
                                for j in range(num_cols_display):
                                    if i + j < len(selected_nums):
                                        data[f"Numérica {j+1}"].append(selected_nums[i+j])
                                    else:
                                        data[f"Numérica {j+1}"].append("")
                                
                                # variable categorica agregada en la ultima columna.
                                if i == 0 and selected_cat:
                                    data["Categórica"].append(selected_cat)

                                else:
                                    data["Categórica"].append("")

                            # se convierte la tabla a un DataFrame para mejor visualizacion.
                            tabla = pd.DataFrame(data)
                            st.table(tabla)

                            # :: inicio GENERACION DE GRAFICOS ::
                            for var in selected_nums:
                                # genera histograma.
                                fig_h = generar_histograma_claro(df, var)
                                if fig_h:
                                    st.plotly_chart(fig_h, use_container_width=True)

                            # genera boxplot.
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

        except Exception as e:
            st.error(f"Error al cargar archivo: {e}")

#:: fin ESTRUCTURA TAB2

#:: inicio ESTRUCTURA TAB1 ::
with tab2:
    st.markdown(
        "<h2 style='text-align: center;'>Asistente Personal - Gemini 1.5</h2>", 
        unsafe_allow_html=True
    )

    df = st.session_state.get('df', None)

    if (df is not None and not df.empty):
        st.markdown(
            "<p style='text-align: center;'>Realiza preguntas a <span style='font-style: italic;'>Gemini</span> para obtener el mejor entendimiento de la información del archivo insertado.</p>",
            unsafe_allow_html=True
        )

        # cuadro de texto donde usuario ingresará la pregunta.
        pregunta = st.text_input("Pregunta para Gemini:", placeholder="¿Qué observas en los datos o en el texto?", key="pregunta_gemini")

        if st.button("Preguntar a Gemini"):
            if df is not None and not df.empty:
                muestra = df.to_csv(index=False)
                prompt = (
                    f"Tengo este dataset (peude ser formato .CSV, .XLS, .XLSX):\n{muestra}\n"
                    f"Pregunta: {pregunta}\nPor favor, responde como si fueras un analista de datos profesional."
                    f"Nunca me des código, necesito que leas todos los datos de todas las columnas a no ser que"
                    f"se te solicite lo contrario. Responde con tablas y bulltepoints si hace falta."
                )
            else:
                st.warning("No hay datos ni texto cargado para enviar a la IA.")
                st.stop()

            try:
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
            "<p style='text-align: center;'>/// Primero sube un documento en la pestaña 'Análisis automático' para habilitar el asistente Gemini. ///</p>",
            unsafe_allow_html=True
        )
# :: fin ESTRUCTURA TAB2 ::

