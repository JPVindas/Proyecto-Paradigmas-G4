import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import PyPDF2
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
import math
from plotly.subplots import make_subplots


# --------------------
# Configuración Streamlit
# --------------------
genai.configure(api_key="AIzaSyB77sw3lhzRhfRrdFMntOhxRLciX9wuuxU")

# Configuración general
genai.configure(api_key="AIzaSyDzzTT-tQLGAFlEVJJx0_Uhir-TbATgVyc")
st.set_page_config(page_title="Análisis Inteligente de Datos", layout="wide")
st.title("🔎 Análisis Automatizado de Datos")
st.markdown("Sube tu archivo **CSV, Excel, PDF, TXT o JSON** y descubre insights al instante.")

# Función de resumen robusta
def resumen_insights(df, num_cols, cat_cols):
    st.set_page_config(page_title="Análisis Inteligente", layout="wide")
    st.title("🔎 Análisis Inteligente de Datos")
    st.markdown("Sube tu archivo **CSV / Excel / JSON / PDF / TXT**. La app hace EDA, detecta outliers, realiza clustering y permite consultar con Gemini (si configuras la API key).")

# --------------------
# UTILIDADES: carga y tipos
# --------------------
@st.cache_data(show_spinner=False, max_entries=5)
def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce el uso de memoria convirtiendo tipos cuando es seguro."""
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['object']).columns:
        # Solo convertir a categoría si no hay demasiadas categorías
        try:
            if df[col].nunique(dropna=True) / max(1, len(df)) < 0.5:
                df[col] = df[col].astype('category')
        except Exception:
            pass
    return df

@st.cache_data(show_spinner=False, max_entries=3)
def cargar_dataframe(archivo) -> pd.DataFrame:
    """Carga CSV/Excel/JSON desde archivo temporal (manejo robusto encoding)."""
    name = archivo.name.lower()
    suffix = os.path.splitext(name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(archivo.getvalue())
        tmp_path = tmp_file.name
    try:
        if name.endswith('.csv'):
            try:
                df = pd.read_csv(tmp_path, low_memory=False)
            except Exception:
                df = pd.read_csv(tmp_path, encoding='latin1', low_memory=False)
        elif name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(tmp_path)
        elif name.endswith('.json'):
            df = pd.read_json(tmp_path)
        else:
            raise ValueError("Formato no soportado por cargar_dataframe")
        return downcast_df(df)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

def extraer_texto_pdf(archivo) -> str:
    """Extrae texto de un PDF (temporal)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(archivo.getvalue())
        tmp_path = tmp.name
    try:
        texto = ""
        with open(tmp_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                texto += p.extract_text() or ""
        return texto
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

@st.cache_data(show_spinner=False, max_entries=10)
def detectar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """Detecta si cada variable es Numérica, Categórica o Temporal."""
    tipo_vars = []
    for col in df.columns:
        col_data = df[col]
        try:
            if pd.api.types.is_datetime64_any_dtype(col_data):
                tipo = "Temporal"
            elif pd.api.types.is_numeric_dtype(col_data):
                nunique = col_data.nunique(dropna=True)
                tipo = "Categórica" if (nunique < 15 or nunique / max(1, len(col_data)) < 0.05) else "Numérica"
            else:
                tipo = "Categórica"
        except Exception:
            tipo = "Categórica"
        tipo_vars.append((col, tipo))
    return pd.DataFrame(tipo_vars, columns=["Variable", "Tipo"])


# ==========================================================
# VISUALIZACIONES CLARAS (Plotly)
# ==========================================================
def _stats_text(serie: pd.Series) -> str:
    """Texto resumen rápido para anotaciones (n, mean, median, std). (ASCII-safe)"""
    n = int(serie.count())
    mean = serie.mean()
    med = serie.median()
    std = serie.std()
    return f"n={n} | mean={mean:.3g} | med={med:.3g} | std={std:.3g}"

def generar_histograma_claro(df: pd.DataFrame, var: str, max_sample=20000):
    """Histograma claro con líneas de media y mediana y estadísticas claras."""
    serie = df[var].dropna()
    if serie.empty:
        return None
    
    sample = serie if len(serie) <= max_sample else serie.sample(max_sample, random_state=42)
    n = len(sample)
    # Ajustar el número de bins para que no sea excesivo en datasets pequeños
    nbins = min(50, max(10, int(math.sqrt(n))))

    mean_val = float(sample.mean())
    median_val = float(sample.median())

    # --- Creación del histograma ---
    fig = px.histogram(
        sample.to_frame(), x=var, nbins=nbins,
        title=f"Histograma de {var} (n={n})",
        labels={var: var.replace('_', ' ').capitalize()} # Mejora de etiqueta del eje x
    )

    # --- Líneas de media y mediana (sin texto aquí) ---
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red")
    fig.add_vline(x=median_val, line_dash="dot", line_color="blue")

    # --- Anotaciones de texto separadas para evitar superposición ---
    # Usamos 'yref="paper"' para posicionar el texto verticalmente
    # en relación al área del gráfico (0=abajo, 1=arriba).
    fig.add_annotation(
        x=mean_val, y=0.95, yref='paper',
        text=f"Media: {mean_val:.2f}",
        showarrow=False,
        font=dict(color="red", size=12),
        xshift=10 # Pequeño desplazamiento horizontal para mejor lectura
    )
    fig.add_annotation(
        x=median_val, y=0.85, yref='paper',
        text=f"Mediana: {median_val:.2f}",
        showarrow=False,
        font=dict(color="blue", size=12),
        xshift=-10 # Pequeño desplazamiento en la otra dirección
    )

    # --- Mejoras de estilo y layout ---
    hover_template = f"{var}: %{{x}}<br>Conteo: %{{y}}<extra></extra>"
    fig.update_traces(
        hovertemplate=hover_template,
        marker_color="lightgreen",
        opacity=0.8
    )

    fig.update_layout(
        yaxis_title="Conteo",
        bargap=0.05,
        title_font_size=18,
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(17, 17, 17, 1)',  # Fondo oscuro para contraste
        paper_bgcolor='rgba(17, 17, 17, 1)',
        font_color='white',
        margin=dict(t=80, b=80, l=60, r=60)
    )

    return fig
def generar_boxplot_claro(df: pd.DataFrame, num_cols: list[str], max_display=6):
    """
    Boxplots horizontales para comparación clara entre variables.
    Excluye variables irrelevantes y normaliza los datos.
    """
    # Excluir la variable 'id' si está presente, ya que no es útil para un boxplot
    filtered_cols = [col for col in num_cols if col.lower() != 'id']
    
    vars_to_plot = filtered_cols[:max_display]
    data = []
    
    # Normalizar los datos para una comparación equitativa
    df_normalized = df[vars_to_plot].dropna()
    if df_normalized.empty:
        return None
        
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_normalized), columns=df_normalized.columns)

    for v in vars_to_plot:
        s = df_scaled[v]
        median_orig = df_normalized[v].median()

        data.append(go.Box(
            x=s,
            name=v,
            boxmean=True,
            marker=dict(opacity=0.7),
            orientation='h',
            hovertemplate=f"<b>Variable: {v}</b><br>Valor Normalizado: %{{x:.2f}}<br>Mediana Original: {median_orig:.2f}<br><extra></extra>"
        ))
    
    if not data:
        return None

    fig = go.Figure(data=data)

    fig.update_layout(
        title="Boxplots Comparativos (Horizontales) - Datos Normalizados",
        xaxis_title="Valor Normalizado",
        yaxis_title="",
        margin=dict(t=60, l=120)
    )

    return fig

def generar_correlacion_clara(df: pd.DataFrame, num_cols: list[str], sample_max=5000):
    """
    Genera una matriz de correlación simplificada o un gráfico de barras de los pares
    más correlacionados, dependiendo de la cantidad de variables.
    """
    if len(num_cols) < 2:
        return None

    sample = df[num_cols].dropna()
    if sample.empty:
        return None
    if len(sample) > sample_max:
        sample = sample.sample(sample_max, random_state=42)
    
    corr = sample.corr()

    # Si hay muchas variables, nos enfocamos en las más relevantes
    if len(num_cols) > 20:
        # Seleccionamos las 10 variables con la correlación absoluta promedio más alta
        mean_abs_corr = corr.abs().mean().sort_values(ascending=False)
        top_10_vars = mean_abs_corr.head(10).index.tolist()
        corr_filtered = corr.loc[top_10_vars, top_10_vars]
        title_text = f"Análisis de Correlación (Top 10 variables más correlacionadas, n={len(sample)})"
    else:
        corr_filtered = corr
        title_text = f"Análisis de Correlación (n={len(sample)})"

    # Máscara para mostrar solo el triángulo inferior
    mask = np.triu(np.ones_like(corr_filtered, dtype=bool))
    corr_masked = corr_filtered.mask(mask)

    # Calcular los pares con mayor correlación
    c_abs = corr.abs().where(~np.eye(len(corr), dtype=bool))
    top_pairs = []
    if not c_abs.empty:
        stacked = c_abs.stack().sort_values(ascending=False)
        for (a, b), val in stacked.items():
            if a != b and (b, a) not in [(p[0], p[1]) for p in top_pairs]:
                top_pairs.append((a, b, val))
            if len(top_pairs) >= 5:
                break
    
    # Crear los subgráficos
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=("Matriz de Correlación (Triángulo Inferior)", "Top 5 Pares Más Correlacionados")
    )

    # Gráfico de calor (heatmap)
    heatmap_fig = px.imshow(
        corr_masked,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1,
    )
    fig.add_trace(heatmap_fig.data[0], row=1, col=1)

    # Gráfico de barras de los pares principales
    if top_pairs:
        labels = [f"{a} <-> {b}" for a, b, val in top_pairs[::-1]]
        values = [val for a, b, val in top_pairs[::-1]]

        bar_fig = go.Figure(
            data=[go.Bar(
                x=values,
                y=labels,
                orientation='h',
                marker_color='lightblue'
            )]
        )
        bar_fig.update_layout(xaxis_title="Correlación Absoluta", yaxis_title="Par de Variables")
        
        fig.add_trace(bar_fig.data[0], row=1, col=2)

    # Actualizar el diseño general
    fig.update_layout(
        title_text=title_text,
        height=600,
        width=1000,
        coloraxis_colorbar=dict(title="Correlación"),
        margin=dict(t=80)
    )

    return fig


def generar_barras_claras(df: pd.DataFrame, var: str, top_n=15):
    """Barras categóricas ordenadas por frecuencia con porcentaje y porcentaje acumulado."""
    if var is None:
        return None
    serie = df[var]
    if pd.api.types.is_categorical_dtype(serie):
        serie = serie.astype(object)
    serie = serie.fillna("<missing>")
    counts = serie.value_counts(dropna=False)
    total = counts.sum()
    counts = counts.sort_values(ascending=False)
    if len(counts) > top_n:
        top = counts.head(top_n)
        other_sum = counts.iloc[top_n:].sum()
        top = pd.concat([top, pd.Series({"<Other>": other_sum})])
        counts = top
    df_counts = counts.reset_index()
    df_counts.columns = [var, "Count"]
    df_counts["Percent"] = (df_counts["Count"] / total * 100).round(2)
    df_counts["CumPercent"] = df_counts["Percent"].cumsum().round(2)
    df_counts["LabelText"] = df_counts["Count"].astype(str) + " (" + df_counts["Percent"].astype(str) + "%)"
    fig = px.bar(df_counts, x=var, y="Count", text="Percent",
                    title=f"Distribucion categorica - {var} (Top {min(top_n, len(df_counts))})",
                    labels={var: var, "Count": "Count"})
    fig.update_traces(hovertemplate="%{x}<br>Count: %{y}<br>Percent: %{text:.2f}%<br>Cum: %{customdata[0]:.2f}%<extra></extra>",
                    customdata=df_counts[["CumPercent"]].to_numpy())
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, uniformtext_minsize=8, margin=dict(t=70))
    return fig

def generar_violin_y_box(df: pd.DataFrame, var: str, sample_max=5000):
    """Violin + Box combinados para ver forma de distribucion y resumen estadistico."""
    serie = df[var].dropna()
    if serie.empty:
        return None
    sample = serie if len(serie) <= sample_max else serie.sample(sample_max, random_state=42)
    mean = float(sample.mean())
    med = float(sample.median())
    fig = go.Figure()
    fig.add_trace(go.Violin(y=sample, name="Distribution", box_visible=True, meanline_visible=False, points='outliers'))
    fig.add_trace(go.Scatter(x=[0.85], y=[mean], mode='markers+text', name='Mean', text=[f"mean={mean:.3g}"], textposition='bottom right', marker=dict(symbol='diamond', size=10)))
    fig.add_trace(go.Scatter(x=[0.85], y=[med], mode='markers+text', name='Median', text=[f"med={med:.3g}"], textposition='top right', marker=dict(symbol='circle', size=8)))
    fig.update_layout(title=f"Violin + Box - {var} (sample n={len(sample)})", yaxis_title=var, showlegend=True, margin=dict(t=70))
    return fig

def plot_clusters_pca(df_vis: pd.DataFrame, explained_ratio):
    """Scatter PCA coloreado por cluster con hover limpio y leyenda de varianza explicada."""
    if df_vis is None or df_vis.empty:
        return None
    ev1 = explained_ratio[0] if len(explained_ratio) > 0 else 0.0
    ev2 = explained_ratio[1] if len(explained_ratio) > 1 else 0.0
    title = f"Clusters visualized (PCA 2D) - Var explained PC1={ev1:.2%}, PC2={ev2:.2%}"
    fig = px.scatter(df_vis, x='PC1', y='PC2', color='Cluster',
                      hover_data=['_orig_index'], title=title)
    fig.update_traces(marker=dict(size=7, opacity=0.8), selector=dict(mode='markers'))
    fig.update_layout(xaxis_title=f"PC1 ({ev1:.2%})", yaxis_title=f"PC2 ({ev2:.2%})", margin=dict(t=80))
    fig.add_annotation(
        x=0.99, y=0.99, xref='paper', yref='paper',
        text="Hover: original index | Colors: clusters",
        showarrow=False, align='right', bgcolor='white', bordercolor='black', borderwidth=1, opacity=0.85
    )
    return fig
# ==========================================================


# ==========================================================
# Detección de outliers (IsolationForest)
# ==========================================================
@st.cache_resource(max_entries=3)
def entrenar_isolation_forest(X: pd.DataFrame, contamination: float):
    """Entrena IsolationForest (con número de estimadores adaptativo)."""
    n_estimators = min(200, max(50, int(np.sqrt(max(1, len(X))))))
    model = IsolationForest(
        n_estimators=n_estimators, contamination=contamination,
        random_state=42, n_jobs=-1
    )
    model.fit(X)
    return model

def detectar_outliers(df: pd.DataFrame, num_cols: list[str], contamination=0.05):
    """Detecta outliers sobre las columnas numéricas indicadas. Devuelve DataFrame de outliers (con índice original)."""
    X = df[num_cols].dropna()
    if X.empty or len(X) < 10:
        return pd.DataFrame()
    sample_size = min(10000, len(X))
    sample = X.sample(sample_size, random_state=42) if len(X) > sample_size else X
    model = entrenar_isolation_forest(sample, contamination)
    preds = model.predict(X)  # 1 normal, -1 outlier
    outliers = X.loc[preds == -1]
    return outliers
# ==========================================================


# ==========================================================
# Clustering (KMeans / MiniBatch + PCA 2D)
# ==========================================================
@st.cache_resource(max_entries=3)
def entrenar_kmeans_model(X: np.ndarray, n_clusters: int, minibatch: bool = True):
    """Entrena KMeans o MiniBatchKMeans según minibatch flag."""
    n_clusters = min(max(2, n_clusters), max(2, X.shape[0] // 2))
    if minibatch or X.shape[0] > 10000:
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=min(1024, max(64, X.shape[0] // 10)),
            random_state=42
        )
    else:
        km = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    km.fit(X)
    return km

def clustering_kmeans(df: pd.DataFrame, num_cols: list[str],
                      n_clusters: int = 3, minibatch: bool = True):
    """Clustering sobre columnas numéricas (estandariza, entrena y devuelve muestra con etiquetas)."""
    X_full = df[num_cols].dropna()
    if X_full.empty or len(X_full) < n_clusters:
        return None, None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full.values)

    train_size = min(20000, X_scaled.shape[0])
    if X_scaled.shape[0] > train_size:
        idx = np.random.RandomState(42).choice(X_scaled.shape[0], train_size, replace=False)
        X_train = X_scaled[idx]
    else:
        X_train = X_scaled

    model = entrenar_kmeans_model(X_train, n_clusters, minibatch)

    vis_size = min(5000, X_scaled.shape[0])
    if X_scaled.shape[0] > vis_size:
        vis_idx = np.random.RandomState(1).choice(X_scaled.shape[0], vis_size, replace=False)
        X_vis = X_scaled[vis_idx]
        df_vis = X_full.iloc[vis_idx].copy()
    else:
        X_vis = X_scaled
        df_vis = X_full.copy()

    labels = model.predict(X_vis)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_vis)

    df_vis = df_vis.reset_index(drop=False).rename(columns={'index': '_orig_index'})
    df_vis['Cluster'] = labels.astype(str)
    df_vis['PC1'] = coords[:, 0]
    df_vis['PC2'] = coords[:, 1]
    explained = pca.explained_variance_ratio_
    return df_vis, (model, scaler, explained)
# ==========================================================


# --------------------
# INTERFAZ PRINCIPAL (Tabs)
# --------------------
tab1, tab2 = st.tabs(["Análisis automático", "🤖 Asistente IA"])

with tab1:
    archivo = st.file_uploader("Sube CSV, Excel, JSON, PDF o TXT", type=["csv", "xlsx", "json", "pdf", "txt"])
    df = None
    texto_extraido = ""

    if archivo:
        name = archivo.name.lower()
        try:
            if name.endswith((".csv", ".xlsx", ".json")):
                with st.spinner("Cargando datos..."):
                    df = cargar_dataframe(archivo)
                    st.success(f"Datos cargados: {len(df)} filas × {len(df.columns)} columnas")
                    st.session_state['df'] = df
                    st.dataframe(df.head(min(10, len(df))))
            elif name.endswith(".pdf"):
                with st.spinner("Extrayendo texto de PDF..."):
                    texto_extraido = extraer_texto_pdf(archivo)
                    st.session_state['texto_extraido'] = texto_extraido
                    st.text_area("Texto extraído (preview)", value=texto_extraido[:3000], height=200)
            else:  # TXT
                try:
                    contenido = archivo.read().decode('utf-8', errors='ignore')
                except Exception:
                    contenido = str(archivo.getvalue())
                st.session_state['texto_extraido'] = contenido
                st.text_area("Contenido (preview)", value=contenido[:3000], height=200)
        except Exception as e:
            st.error(f"Error al cargar archivo: {e}")

    # Si no se sube archivo, comprobar session_state
    df = st.session_state.get('df', None)
    texto_extraido = st.session_state.get('texto_extraido', "")

    if df is not None and not df.empty:
        # Detectar tipos
        with st.spinner("Analizando estructura de datos..."):
            tipo_df = detectar_tipos(df)
        st.subheader("📊 Tipos de datos detectados")
        st.dataframe(tipo_df, use_container_width=True)

        # Columnas numéricas y categóricas
        num_cols = tipo_df[tipo_df["Tipo"] == "Numérica"]["Variable"].tolist()
        cat_cols = tipo_df[tipo_df["Tipo"] == "Categórica"]["Variable"].tolist()

        st.subheader("📈 Estadísticas descriptivas (numéricas)")
        try:
            st.dataframe(df[num_cols].describe(), use_container_width=True)
        except Exception:
            st.dataframe(df.describe(include='all'), use_container_width=True)

        st.subheader("🔍 Visualizaciones mejoradas y claras")

        # Elegir variables representativas para mostrar (si hay muchas, seleccionamos top por varianza)
        if num_cols:
            var_varianza = df[num_cols].var().sort_values(ascending=False)
            selected_nums = var_varianza.index[:min(3, len(var_varianza))].tolist()
        else:
            selected_nums = []

        # Para categórica, usar la que tenga más valores no nulos y <= 50 categorías (si existe)
        selected_cat = None
        best_nonnull = -1
        for c in cat_cols:
            non_null = df[c].notna().sum()
            unique_ct = df[c].nunique(dropna=True)
            if unique_ct <= 50 and non_null > best_nonnull:
                selected_cat = c
                best_nonnull = non_null

        # Mostrar qué variables se han seleccionado automáticamente
        st.markdown("**Variables usadas en las gráficas (selección automática):**")
        cols_info_left, cols_info_right = st.columns(2)
        with cols_info_left:
            st.write("Numéricas (por varianza):")
            if selected_nums:
                for v in selected_nums:
                    st.write(f"- {v} (var={df[v].var():.4g})")
            else:
                st.write("- (no hay variables numéricas detectadas)")
        with cols_info_right:
            st.write("Categórica seleccionada:")
            st.write(f"- {selected_cat}" if selected_cat else "- (no se encontró categórica adecuada)")

        # 1) Histogramas individuales (para cada selected_num)
        for var in selected_nums:
            fig_h = generar_histograma_claro(df, var)
            if fig_h:
                st.plotly_chart(fig_h, use_container_width=True)

        # 2) Boxplots comparativos
        fig_box = generar_boxplot_claro(df, selected_nums, max_display=6)
        if fig_box:
            st.plotly_chart(fig_box, use_container_width=True)

        # 3) Correlación (si hay al menos 2 numéricas)
        fig_corr = generar_correlacion_clara(df, num_cols)
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True)

        # 4) Barras categóricas (clara)
        if selected_cat:
            fig_bar = generar_barras_claras(df, selected_cat, top_n=20)
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)

        # 5) Violin + Box para la primera variable numérica (si existe)
        if selected_nums:
            fig_vb = generar_violin_y_box(df, selected_nums[0])
            if fig_vb:
                st.plotly_chart(fig_vb, use_container_width=True)

        # --------------------
        # Análisis avanzado: Outliers
        # --------------------
        st.subheader("⚠️ Detección de outliers (IsolationForest)")
        if selected_nums and len(selected_nums) >= 1:
            contamination = st.slider("Sensibilidad outliers (contamination)", 0.01, 0.2, 0.05, 0.01, key="contamination_slider")
            with st.spinner("Detectando outliers..."):
                outliers = detectar_outliers(df, selected_nums, contamination=contamination)
            if not outliers.empty:
                frac = len(outliers) / max(1, len(df))
                st.info(f"Outliers detectados: {len(outliers)} — {frac:.2%} del dataset (según las variables usadas)")
                st.dataframe(outliers.head(200))
            else:
                st.success("No se detectaron outliers con los parámetros actuales.")
        else:
            st.info("Se requieren columnas numéricas para detección de outliers.")

        # --------------------
        # Análisis avanzado: Clustering
        # --------------------
        st.subheader("🧩 Clustering (KMeans / MiniBatchKMeans) — visualización PCA 2D")
        if num_cols and len(num_cols) >= 2:
            n_clusters = st.slider("Número de clusters", 2, 12, 3, key="nclusters_slider")
            use_minibatch = st.checkbox("Usar MiniBatchKMeans (recomendado para datasets grandes)", value=True, key="use_minibatch")
            with st.spinner("Calculando clusters (esto puede tardar en datasets grandes)..."):
                var_var = df[num_cols].var().sort_values(ascending=False)
                cluster_cols = var_var.index[:min(12, len(var_var))].tolist()  # limitar a 12 columnas para estabilidad
                df_clusters_vis, model_info = clustering_kmeans(df, cluster_cols, n_clusters=n_clusters, minibatch=use_minibatch)
            if df_clusters_vis is not None:
                st.info(f"Muestra de {len(df_clusters_vis)} observaciones usadas para visualizar clusters")
                st.dataframe(df_clusters_vis.head(100))
                fig_clusters = plot_clusters_pca(df_clusters_vis, model_info[2])
                if fig_clusters:
                    st.plotly_chart(fig_clusters, use_container_width=True)
            else:
                st.warning("No fue posible realizar clustering con las columnas seleccionadas (datos insuficientes).")
        else:
            st.info("Se requieren al menos 2 columnas numéricas para clustering.")

    else:
        st.info("Carga un archivo (CSV/Excel/JSON) para generar análisis y visualizaciones.")




# ==========================================================
# ======== Tab 1: Análisis Automático =========

with tab1:
    st.write("Carga un archivo (CSV/Excel/JSON) para generar análisis y visualizaciones.")

# ==========================================================
# ======== Tab 2: Asistente IA Gemini =========

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
                    f"Tengo el siguiente texto extraído de un archivo (puede ser PDF, TXT, etc):\n"
                    f"{texto_extraido[:3000]}\n"
                    f"Pregunta: {pregunta}\nPor favor, responde de manera concisa y profesional."
                )
            else:
                st.warning("No hay datos ni texto cargado para enviar a la IA.")
                st.stop()

            try:
                modelo = genai.GenerativeModel('models/gemini-1.5-flash-latest')
                respuesta = modelo.generate_content(prompt)
                st.write("**Respuesta de Gemini:**")
                st.success(respuesta.text)
            except Exception as e:
                st.error(f"Error al comunicarse con Gemini: {e}")
    else:
        st.info("Primero sube un archivo de datos o texto en la pestaña 'Análisis automático' para habilitar el asistente Gemini.")
