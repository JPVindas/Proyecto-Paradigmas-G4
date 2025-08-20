import streamlit as st
from pathlib import Path

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
            st.switch_page("pages/analisis_de_celdas.py")
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
            st.switch_page("pages/lector_de_documentos.py")
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
            st.switch_page("pages/lector_de_imagenes.py")
        st.markdown(
            f"""
            <div class="card-container">
                <div class="card-icon">🖼️</div>
                <div class="card-title">Lector de Imágenes</div>
                <div class="card-description">Extrae texto y objetos de imágenes.</div>
            </div>
            """, unsafe_allow_html=True
        )