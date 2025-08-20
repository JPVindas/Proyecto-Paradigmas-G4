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

from matplotlib.colors import to_rgb


# --------------------
# Configuración Streamlit
# --------------------


# Configuración general
genai.configure(api_key="AIzaSyDzzTT-tQLGAFlEVJJx0_Uhir-TbATgVyc")
st.set_page_config(page_title="Análisis Inteligente de Datos", layout="wide")

# --- Page configuration ---
st.set_page_config(
    page_title="Data Analyzer Pro",
    layout="wide"
)

# --- CSS for card design ---
st.markdown("""
<style>
.card-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 20px;
    background-color: #f0f2f6; /* light background */
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
}
.card-container:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
}
.card-icon {
    font-size: 48px;
    margin-bottom: 10px;
}
.card-title {
    color: #4A90E2; /* blue */
    font-size: 1.5rem;
    font-weight: bold;
    margin-bottom: 5px;
}
.card-description {
    color: #555;
    font-size: 14px;
}
.stButton>button {
    width: 100%;
    border: none;
    background: none;
    padding: 0;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: center;'>Bienvenido a Data Analyzer Pro</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; margin: 0;'>En esta plataforma podrás importar archivos para generar un análisis gráfico de la información.</p>" \
    "<p style='text-align: center;'>Además, podrás consultar con una motor de Inteligencia Artificial Generativa para entender la información a tu manera.</p>", 
    unsafe_allow_html=True
)

# Card-based button columns
col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        if st.button("Análisis de Celdas", key="csv_btn"):
            st.switch_page("pages/csv_reader.py")
        st.markdown(
            f"""
            <div class="card-container">
                <div class="card-icon">📄</div>
                <div class="card-title">Análisis de Celdas</div>
                <div class="card-description">Analiza y visualiza datos tabulares.</div>
            </div>
            """, unsafe_allow_html=True
        )

with col2:
    with st.container():
        if st.button("Lector de Documentos", key="pdf_btn"):
            st.switch_page("pages/pdf_reader.py")
        st.markdown(
            f"""
            <div class="card-container">
                <div class="card-icon">📚</div>
                <div class="card-title">Lector de Documentos</div>
                <div class="card-description">Extrae texto y datos de documentos PDF.</div>
            </div>
            """, unsafe_allow_html=True
        )

with col3:
    with st.container():
        if st.button("Lector de Imágenes", key="png_btn"):
            st.switch_page("pages/png_reader.py")
        st.markdown(
            f"""
            <div class="card-container">
                <div class="card-icon">🖼️</div>
                <div class="card-title">Lector de Imágenes</div>
                <div class="card-description">Extrae texto y objetos de imágenes.</div>
            </div>
            """, unsafe_allow_html=True
        )

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
# metodo para cargar del DataFrame (recibe el archivo subido)
def cargar_dataframe(archivo) -> pd.DataFrame: 

    # obtiene nombre de archivo y lo pasa a minusculas.
    name = archivo.name.lower()
    suffix = os.path.splitext(name)[1]

    # genera archivo temporal (archivo subido clonado) en disco con para facilitar lectura.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(archivo.getvalue())
        tmp_path = tmp_file.name

    try:
        if name.endswith('.csv'):
            try:
                # lectura de archivo .CSV con libreria 'pandas'.
                df = pd.read_csv(tmp_path, low_memory=False)
            except Exception:
                # diferente metodo de lectura usado para resolucion de problemas de caracteres
                df = pd.read_csv(tmp_path, encoding='latin1', low_memory=False)

        elif name.endswith(('.xls', '.xlsx')):
            # lectura de archivo .XLS / .XLSX con libreria 'pandas'.
            df = pd.read_excel(tmp_path)

        else:
            raise ValueError("Formato no soportado.")

        # retorna el DataFrame    
        return downcast_df(df)

    finally:
        try:
            # borra archivo temporal
            os.unlink(tmp_path)
        except Exception:
            pass

@st.cache_data(show_spinner=False, max_entries=10)
# metodo para detectar tipos de datos en el DataFrame (recibe el DataFrame)
def detectar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    # lista vacia que almacena los tipos.
    tipo_vars = []

    for col in df.columns:
        col_data = df[col]
        try:
            # "si la columna es tipo date, datetime..."
            if pd.api.types.is_datetime64_any_dtype(col_data):
                tipo = "Temporal"
            # "si la columna es tipo int, float, ..."
            elif pd.api.types.is_numeric_dtype(col_data):
                tipo = "Numérica"
            # "de lo contrario..."
            else:
                tipo = "Categórica"
        except Exception:
            tipo = "Categórica"

        # lista = [nombre_columna | tipo_columna]
        tipo_vars.append((col, tipo))

    # retorna ud DataFrame con la lista de [tipo_vars] como tabla.  
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

# funcion que genera histograma (recibe el DataFrame y las variables numericas seleccionadas).
def generar_histograma_claro(df: pd.DataFrame, var: str, max_sample=20000):
    # extrae columna numerica y elimina nulos en caso de tenerlos.
    serie = df[var].dropna()
    if serie.empty:
        return None
    
    # "si la muestra es <= 20000, se usa completa, de lo contrario se hace una muestra aleatoria de ese tamanho".
    sample = serie if len(serie) <= max_sample else serie.sample(max_sample, random_state=42)

    # limita numero de bins para mejor manejo de Datasets pequenhos.
    n = len(sample)
    nbins = min(50, max(10, int(math.sqrt(n))))

    mean_val = float(sample.mean()) # calcula media.
    median_val = float(sample.median()) # calcula mediana.

    # :: inicio GENERACION DEL HISTOGRAMA ::
    st.markdown(
        f"<h3 style='text-align: center;'>Histograma de {var}</h3>",
        unsafe_allow_html=True
    )

    fig = px.histogram(
        # [x] = columna a graficar
        sample.to_frame(), x=var, nbins=nbins,
        labels={var: var.replace('_', ' ').capitalize()} # Mejora de etiqueta del eje x
    )

    # lineas de media y mediana
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red")
    fig.add_vline(x=median_val, line_dash="dot", line_color="blue")

    fig.add_annotation(
        x=mean_val, y=0.95, yref='paper',
        text=f"Media: {mean_val:.2f}",
        showarrow=False,
        font=dict(color="red", size=12),
        xshift=10
    )
    fig.add_annotation(
        x=median_val, y=0.85, yref='paper',
        text=f"Mediana: {median_val:.2f}",
        showarrow=False,
        font=dict(color="blue", size=12),
        xshift=-10
    )

    # :: inicio estilos de histograma ::
    counts, bins = np.histogram(sample, bins=nbins)
    norm_counts = counts / counts.max()
    start_color = np.array(to_rgb("#3C91E6"))
    end_color = np.array(to_rgb("#7CBEFF"))

    colors = [
        f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
        for r,g,b in (start_color + t*(end_color - start_color) for t in norm_counts)
    ]

    hover_template = f"{var}: %{{x}}<br>Conteo: %{{y}}<extra></extra>"
    fig.update_traces(
        marker_color=colors, 
        opacity=0.8, 
        hovertemplate=hover_template
    )

    fig.update_layout(
        title=f"",
        yaxis_title="Conteo",
        bargap=0.05,
        title_font_size=18,
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='black',
        margin=dict(l=20, r=20, t=30, b=20)
    )
    # :: fin estilos de histograma ::

    # :: fin GENERACION DEL HISTOGRAMA ::

    return fig

# funcion que genera box-lot (recibe el DataFrame y las variables numericas seleccionadas).
def generar_boxplot_claro(df: pd.DataFrame, num_cols: list[str], max_display=6):
    filtered_cols = [col for col in num_cols if col.lower() != 'id'] # excluye variable id si hay presente.
    
    # toma las primeras 6 columnas numericas ya filtradas.
    vars_to_plot = filtered_cols[:max_display]
    data = []
    
    # selecciona las columnas a graficar y elimina nulos.
    df_normalized = df[vars_to_plot].dropna()
    if df_normalized.empty:
        return None

    # crea un nuevo DataFrame con los datos normalizados (media 0, desviacion estandar 1).      
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_normalized), columns=df_normalized.columns)

    # :: inicio GENERACION DEL BOXLPOT ::

    # recorre las columnas almacenadaos en la variable que se van a graficar.
    for v in vars_to_plot:
        s = df_scaled[v]
        median_orig = df_normalized[v].median()

        # almacena boxplot para cada variable recorrida.
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

    # genera la figura con los boxplots guardados con la libreria 'Plotly'.
    fig = go.Figure(data=data)

    # estilos de boxplot
    fig.update_layout(
        title="Boxplots Comparativos (Horizontales) - Datos Normalizados",
        xaxis_title="Valor Normalizado",
        yaxis_title="",
        margin=dict(t=60, l=120)
    )

    # :: fin GENERACION DEL BOXLPOT ::

    return fig

# funcion que genera diagrama de correlacion (recibe el DataFrame y las variables numericas seleccionadas).
def generar_correlacion_clara(df: pd.DataFrame, num_cols: list[str], sample_max=5000):
    # si hay menos de 2 columas numericas no se puede hacer correlacion.
    if len(num_cols) < 2:
        return None

    # extrae muestra de columnas numericas y elimina nulos en caso de tenerlos.
    sample = df[num_cols].dropna()
    if sample.empty:
        return None
    
    # si la muestra es mayor a 5000 inserciones, se hace una muestra aleatoria.
    if len(sample) > sample_max:
        sample = sample.sample(sample_max, random_state=42)
    
    # calcula matriz de correlacion entre las columnas de la muestra.
    corr = sample.corr()

    # "si hay mas de 20 columnas numericas..."
    if len(num_cols) > 20:
        # calcula la media de los valores absolutos de la correlacion de la variable y los ordena de forma descendente.
        mean_abs_corr = corr.abs().mean().sort_values(ascending=False)

        # se seleccionan las 10 más correlacionables en promedio.
        top_10_vars = mean_abs_corr.head(10).index.tolist()

        # reduce la matriz a las 10 variables seleccionadas.
        corr_filtered = corr.loc[top_10_vars, top_10_vars]

        title_text = f"Análisis de Correlación (Top 10 variables más correlacionadas, n={len(sample)})"
    else:
        corr_filtered = corr
        title_text = f"Análisis de Correlación (n={len(sample)})"

    # almacena mascara con forma de matriz de correlacion de traingulo inferior para no duplicar info.
    mask = np.triu(np.ones_like(corr_filtered, dtype=bool))
    corr_masked = corr_filtered.mask(mask)

    # calcula matriz de correlacion en valor absoluto (excluye redundancia).
    c_abs = corr.abs().where(~np.eye(len(corr), dtype=bool))

    # lista donde se guardan los pares con mayor correlacion.
    top_pairs = []
    if not c_abs.empty:
        stacked = c_abs.stack().sort_values(ascending=False)
        for (a, b), val in stacked.items():
            if a != b and (b, a) not in [(p[0], p[1]) for p in top_pairs]:
                top_pairs.append((a, b, val))
            if len(top_pairs) >= 5:
                break
    
    # creacion de subgraficos.
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=("Matriz de Correlación (Triángulo Inferior)", "Top 5 Pares Más Correlacionados")
    )

    # grafico de heatmap.
    heatmap_fig = px.imshow(
        corr_masked,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1,
    )
    fig.add_trace(heatmap_fig.data[0], row=1, col=1)

    # grafico de barras de los pares principales
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

    # disenho layout
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