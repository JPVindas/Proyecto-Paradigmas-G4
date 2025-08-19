# leer_json_sql.py
import streamlit as st
import pandas as pd
import re
import google.generativeai as genai
from app import (
    cargar_dataframe,           # solo si hay JSON tabular
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
# Configuración de página
# =========================
st.set_page_config(page_title="📊 Analizador JSON/SQL", layout="wide")
genai.configure(api_key="AIzaSyDzzTT-tQLGAFlEVJJx0_Uhir-TbATgVyc")  # API Gemini

st.title("📊 Análisis de Archivos JSON y SQL")
st.markdown("Sube archivos JSON para análisis tabular o SQL para analizar tablas y estructura de datos.")

# =========================
# Función para seleccionar variables representativas
# =========================
def seleccionar_variables(df, num_cols, cat_cols):
    selected_nums = df[num_cols].var().sort_values(ascending=False).index[:3].tolist() if num_cols else []
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
# Funciones para SQL
# =========================
def analizar_sql(texto_sql):
    """
    Extrae tablas y columnas de CREATE TABLE y cuenta filas de INSERT INTO.
    """
    tablas = {}
    # Detectar CREATE TABLE
    pattern_create = re.compile(r"CREATE\s+TABLE\s+`?(\w+)`?\s*\((.*?)\);", re.IGNORECASE | re.DOTALL)
    for match in pattern_create.finditer(texto_sql):
        nombre = match.group(1)
        columnas_raw = match.group(2)
        columnas = parse_columnas_create_table(columnas_raw)
        tablas[nombre] = {"columnas": columnas, "insert_count": 0}

    # Contar INSERT INTO
    pattern_insert = re.compile(r"INSERT\s+INTO\s+`?(\w+)`?", re.IGNORECASE)
    for match in pattern_insert.finditer(texto_sql):
        nombre = match.group(1)
        if nombre in tablas:
            tablas[nombre]["insert_count"] += 1

    return tablas

def parse_columnas_create_table(columnas_raw):
    """Extrae nombres de columnas válidos de CREATE TABLE evitando claves y duplicados"""
    columnas = []
    for linea in columnas_raw.split(","):
        linea = linea.strip()
        if not linea or linea.upper().startswith(("PRIMARY", "FOREIGN", "UNIQUE", "CONSTRAINT", ")")):
            continue
        # Tomar solo el primer "token" como nombre de columna
        col_name = linea.split()[0].replace("`", "").strip()
        columnas.append(col_name)
    # Eliminar duplicados manteniendo orden
    seen = set()
    columnas_unicas = []
    for c in columnas:
        if c not in seen:
            columnas_unicas.append(c)
            seen.add(c)
    return columnas_unicas

def parse_insert_values(texto_sql):
    """
    Extrae los valores de cada INSERT INTO y los organiza por tabla.
    Retorna diccionario: {tabla: [fila1, fila2, ...]}
    """
    inserts = {}
    # Match INSERT INTO nombre_tabla (...) VALUES (...)
    pattern_insert = re.compile(
        r"INSERT\s+INTO\s+`?(\w+)`?\s*\((.*?)\)\s*VALUES\s*(.*?);",
        re.IGNORECASE | re.DOTALL
    )
    for match in pattern_insert.finditer(texto_sql):
        tabla = match.group(1)
        columnas = [c.strip().replace("`","") for c in match.group(2).split(",")]
        valores_raw = match.group(3).strip()
        # Separar cada fila (entre paréntesis)
        filas = re.findall(r"\((.*?)\)", valores_raw, re.DOTALL)
        filas_parseadas = []
        for f in filas:
            # Separar por comas, respetando strings entre comillas
            row = [v.strip().strip("'").strip('"') for v in re.split(r",(?=(?:[^']*'[^']*')*[^']*$)", f)]
            filas_parseadas.append(dict(zip(columnas, row)))
        if tabla not in inserts:
            inserts[tabla] = []
        inserts[tabla].extend(filas_parseadas)
    return inserts

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["Análisis Automático", "🤖 Asistente IA"])

# =========================
# Tab 1: Análisis Automático
# =========================
with tab1:
    archivo = st.file_uploader("Sube JSON o SQL", type=["json", "sql"])
    df = None
    texto_sql = ""

    if archivo:
        name = archivo.name.lower()
        try:
            if name.endswith(".json"):
                # Cargar JSON robusto
                try:
                    df = pd.read_json(archivo)
                except ValueError:
                    archivo.seek(0)
                    df = pd.read_json(archivo, orient="records")
                st.session_state['df'] = df
                st.success(f"JSON cargado: {len(df)} filas × {len(df.columns)} columnas")
                st.dataframe(df.head(10))

            elif name.endswith(".sql"):
                texto_sql = archivo.read().decode("utf-8", errors="ignore")
                st.session_state['texto_sql'] = texto_sql
                st.code(texto_sql[:5000], language="sql")

        except Exception as e:
            st.error(f"No se pudo cargar el archivo: {e}")

    # Recuperar session_state
    df = st.session_state.get('df', None)
    texto_sql = st.session_state.get('texto_sql', "")

    # =========================
    # Análisis automático JSON
    # =========================
    if df is not None and not df.empty:
        try:
            tipo_df = detectar_tipos(df)
            st.subheader("📊 Tipos de datos detectados")
            st.dataframe(tipo_df, use_container_width=True)

            num_cols = tipo_df[tipo_df["Tipo"] == "Numérica"]["Variable"].tolist()
            cat_cols = tipo_df[tipo_df["Tipo"] == "Categórica"]["Variable"].tolist()

            st.subheader("📈 Estadísticas descriptivas (numéricas)")
            st.dataframe(df[num_cols].describe(), use_container_width=True)

            st.subheader("🔍 Visualizaciones automáticas")
            selected_nums, selected_cat = seleccionar_variables(df, num_cols, cat_cols)

            # Mostrar variables seleccionadas
            cols_info_left, cols_info_right = st.columns(2)
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

            # Gráficos
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

            # Outliers
            if selected_nums:
                contamination = st.slider("Sensibilidad outliers", 0.01, 0.2, 0.05, 0.01)
                outliers = detectar_outliers(df, selected_nums, contamination)
                if not outliers.empty:
                    st.info(f"Outliers detectados: {len(outliers)} — {len(outliers)/len(df):.2%}")
                    st.dataframe(outliers.head(200))

            # Clustering
            if num_cols and len(num_cols) >= 2:
                n_clusters = st.slider("Número de clusters", 2, 12, 3)
                df_clusters_vis, model_info = clustering_kmeans(df, num_cols, n_clusters=n_clusters, minibatch=True)
                if df_clusters_vis is not None:
                    st.dataframe(df_clusters_vis.head(100))
                    fig_clusters = plot_clusters_pca(df_clusters_vis, model_info[2])
                    if fig_clusters:
                        st.plotly_chart(fig_clusters, use_container_width=True)

        except Exception as e:
            st.error(f"Error al analizar JSON: {e}")

    # =========================
    # Análisis SQL
    # =========================
    if texto_sql:
        st.subheader("📋 Resumen de tablas y columnas en SQL")
        tablas = analizar_sql(texto_sql)
        inserts = parse_insert_values(texto_sql)
        for nombre, info in tablas.items():
            st.markdown(f"**Tabla `{nombre}`**")
            if info['columnas']:
                # Mostrar tabla ilustrativa con datos insertados si existen
                if nombre in inserts and inserts[nombre]:
                    df_demo = pd.DataFrame(inserts[nombre])
                else:
                    df_demo = pd.DataFrame(columns=info['columnas'])
                st.dataframe(df_demo)
            else:
                st.write("No se detectaron columnas válidas")
            st.write(f"Cantidad de INSERT INTO detectados: {info['insert_count']}")

# =========================
# Tab 2: Asistente IA
# =========================
with tab2:
    st.markdown("## 🤖 Asistente Inteligente (Gemini)")
    pregunta = st.text_area("Pregunta:", placeholder="Hazle preguntas sobre tu JSON o SQL...")

    if st.button("Preguntar a Gemini"):
        df = st.session_state.get('df', None)
        texto_sql = st.session_state.get('texto_sql', "")

        if df is not None and not df.empty:
            muestra = df.head(10).to_csv(index=False)
            prompt = f"Tengo este JSON:\n{muestra}\nPregunta: {pregunta}"
        elif texto_sql:
            prompt = f"Tengo este código SQL:\n{texto_sql[:5000]}\nPregunta: {pregunta}\nAdemás, dame un resumen de las tablas, columnas y recomendaciones."
        else:
            st.warning("Sube un archivo primero")
            st.stop()

        try:
            modelo = genai.GenerativeModel('gemini-1.5-flash')
            respuesta = modelo.generate_content(prompt)
            st.write("**Respuesta de Gemini:**")
            st.success(respuesta.text)
        except Exception as e:
            st.error(f"Error al comunicarse con Gemini: {e}")
