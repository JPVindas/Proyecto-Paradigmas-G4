import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import plotly.express as px
import google.generativeai as genai

# ======= Configuración general =======
genai.configure(api_key="AIzaSyB77sw3lhzRhfRrdFMntOhxRLciX9wuuxU")
st.set_page_config(page_title="Análisis Inteligente de Datos", layout="wide")
st.title("🔎 Análisis Automatizado de Datos")
st.markdown("Sube tu archivo **CSV o Excel** y descubre insights al instante.")

# ======= Función de resumen robusta =======
def resumen_insights(df, num_cols, cat_cols):
    st.markdown("### 📝 **Resumen rápido de insights**")
    if not num_cols:
        st.info("- No hay variables numéricas para análisis estadístico profundo.")
    else:
        for col in num_cols:
            st.write(
                f"- **{col}:** Media={df[col].mean():.2f}, "
                f"Mín={df[col].min()}, Máx={df[col].max()}, Nulos={df[col].isnull().sum()}"
            )
    if not cat_cols:
        st.info("- No hay variables categóricas para análisis de frecuencia.")
    else:
        for col in cat_cols:
            mode = df[col].mode(dropna=True)
            top = mode.iloc[0] if not mode.empty else "Sin valor dominante"
            st.write(f"- **{col}:** {df[col].nunique()} categorías únicas, Top: {top}")

# ======= Funciones utilitarias =======
def detectar_tipos(df):
    tipo_vars = {}
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            tipo_vars[col] = "Categórica" if df[col].nunique() < 15 else "Numérica"
        elif np.issubdtype(df[col].dtype, np.datetime64):
            tipo_vars[col] = "Temporal"
        else:
            tipo_vars[col] = "Categórica"
    return pd.DataFrame(list(tipo_vars.items()), columns=["Variable", "Tipo"])

def mostrar_correlacion(df, num_cols):
    if len(num_cols) < 2:
        st.info("No hay suficientes variables numéricas para matriz de correlación.")
        return
    corr = df[num_cols].corr()
    if corr.isnull().values.all():
        st.warning("No hay datos suficientes para calcular correlación.")
        return
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

def mostrar_histograma(df, var):
    if df[var].dropna().empty:
        st.warning(f"No hay datos para graficar el histograma de: {var}")
        return
    fig, ax = plt.subplots()
    sns.histplot(df[var].dropna(), kde=True, ax=ax)
    st.pyplot(fig)

def mostrar_barras(df, var):
    conteo = df[var].value_counts()
    if conteo.empty:
        st.warning(f"No hay datos para graficar la variable categórica: {var}")
        return
    fig, ax = plt.subplots()
    conteo.plot(kind='bar', ax=ax)
    plt.title(var)
    st.pyplot(fig)

def detectar_outliers(df, num_cols):
    if not num_cols:
        return pd.DataFrame()
    model = IsolationForest(contamination=0.05)
    X = df[num_cols].dropna()
    if X.empty:
        return pd.DataFrame()
    outlier_pred = model.fit_predict(X)
    df_out = df.loc[X.index].copy()
    df_out['Es_outlier'] = outlier_pred == -1
    return df_out[df_out['Es_outlier']]

def clustering_kmeans(df, num_cols, n_clusters):
    X = df[num_cols].dropna()
    if X.empty:
        return None
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    labels = kmeans.fit_predict(X)
    df_clust = df.copy()
    df_clust['Cluster'] = -1
    df_clust.loc[X.index, 'Cluster'] = labels
    return df_clust

# ========== Tabs: Analítica y Asistente IA ==========
tab1, tab2 = st.tabs(["Análisis automático", "🤖 Asistente IA (Gemini)"])

# ======== Tab 1: Analítica tradicional =========
with tab1:
    archivo = st.file_uploader("Selecciona un archivo CSV o Excel", type=["csv", "xlsx"], key="fileuploader")
    df = None

    # Cargar y guardar DataFrame en session_state
    if archivo:
        try:
            if archivo.name.endswith(".csv"):
                df = pd.read_csv(archivo)
            elif archivo.name.endswith(".xlsx"):
                df = pd.read_excel(archivo)
            st.success("¡Archivo cargado exitosamente!")
            st.write("### Vista previa de los datos", df.head())
            st.session_state['df'] = df
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
    elif 'df' in st.session_state:
        df = st.session_state['df']

    if df is not None and not df.empty:
        tipo_df = detectar_tipos(df)
        st.write("#### Detección automática de variables:")
        st.dataframe(tipo_df)

        st.write("### Estadísticas Descriptivas (Variables Numéricas):")
        st.dataframe(df.describe())

        num_cols = tipo_df[tipo_df["Tipo"] == "Numérica"]["Variable"].tolist()
        cat_cols = tipo_df[tipo_df["Tipo"] == "Categórica"]["Variable"].tolist()

        # --- Resumen robusto aquí ---
        resumen_insights(df, num_cols, cat_cols)

        st.write("### Matriz de correlación:")
        mostrar_correlacion(df, num_cols)

        st.write("### Visualizaciones automáticas:")
        col1, col2 = st.columns(2)
        with col1:
            if num_cols:
                var = st.selectbox("Selecciona variable numérica para histograma", num_cols)
                mostrar_histograma(df, var)
        with col2:
            if cat_cols:
                var_cat = st.selectbox("Selecciona variable categórica para gráfico de barras", cat_cols)
                mostrar_barras(df, var_cat)

        st.write("### Detección de outliers (Isolation Forest):")
        outliers = detectar_outliers(df, num_cols)
        if outliers is not None and not outliers.empty:
            st.write(f"Se detectaron **{len(outliers)}** valores atípicos en variables numéricas.")
            st.dataframe(outliers)
        else:
            st.info("No hay valores atípicos detectados o no hay suficientes datos.")

        st.write("### Clustering automático (KMeans):")
        if len(num_cols) >= 2:
            n_clusters = st.slider("Selecciona cantidad de clusters", min_value=2, max_value=5, value=3)
            df_clusters = clustering_kmeans(df, num_cols, n_clusters)
            if df_clusters is not None and 'Cluster' in df_clusters:
                st.write("Agrupamientos detectados por KMeans:")
                st.dataframe(df_clusters[['Cluster'] + num_cols].sort_values("Cluster"))
                st.write("Visualización 2D de los dos primeros atributos numéricos:")
                fig = px.scatter(df_clusters, x=num_cols[0], y=num_cols[1], color=df_clusters['Cluster'].astype(str),
                                 title="Clustering KMeans")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay suficientes datos válidos para realizar clustering.")
        else:
            st.info("Se requieren al menos dos variables numéricas para clustering.")

        st.write("---")
        st.write("¿Quieres descargar el resumen?")
        resumen_csv = df.describe().to_csv().encode('utf-8')
        st.download_button("Descargar resumen CSV", data=resumen_csv, file_name="resumen_estadistico.csv")
    else:
        st.info("Sube un archivo de datos para ver análisis.")

# ======== Tab 2: Asistente IA Gemini =========
with tab2:
    st.markdown("## 🤖 Asistente Inteligente (Gemini)")
    df = st.session_state.get('df', None)
    if df is not None and not df.empty:
        st.write("Hazle preguntas a la IA sobre tu conjunto de datos (máx 10 primeras filas enviadas). Por ejemplo: '¿Qué variables parecen estar más relacionadas?' o '¿Hay patrones curiosos?'")
        pregunta = st.text_area("Pregunta para Gemini:", placeholder="¿Qué observas en los datos?", key="pregunta_gemini")
        if st.button("Preguntar a Gemini"):
            muestra = df.head(10).to_csv(index=False)
            prompt = (
                f"Tengo este dataset en formato CSV:\n{muestra}\n"
                f"Pregunta: {pregunta}\nPor favor, responde como si fueras un analista de datos profesional."
            )
            try:
                modelo = genai.GenerativeModel('models/gemini-1.5-flash-latest')
                respuesta = modelo.generate_content(prompt)
                st.write("**Respuesta de Gemini:**")
                st.success(respuesta.text)
            except Exception as e:
                st.error(f"Error al comunicarse con Gemini: {e}")
    else:
        st.info("Primero sube un archivo de datos en la pestaña 'Análisis automático' para habilitar el asistente Gemini.")
