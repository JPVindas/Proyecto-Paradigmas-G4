import streamlit as st
from pathlib import Path

# --- Page configuration ---
st.set_page_config(
    page_title="Data Analyzer Pro",
    page_icon="✨",
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

# --- Page Content ---
st.title("✨ Bienvenido a Data Analyzer Pro")
st.markdown("### Elige una opción para comenzar a analizar tus datos.")
st.write("") # Spacer

# Card-based button columns
col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        if st.button("CSV Reader", key="csv_btn"):
            st.switch_page("pages/csv_reader.py")
        st.markdown(
            f"""
            <div class="card-container">
                <div class="card-icon">📄</div>
                <div class="card-title">Lector de CSV</div>
                <div class="card-description">Analiza y visualiza datos tabulares.</div>
            </div>
            """, unsafe_allow_html=True
        )

with col2:
    with st.container():
        if st.button("PDF Reader", key="pdf_btn"):
            st.switch_page("pages/pdf_reader.py")
        st.markdown(
            f"""
            <div class="card-container">
                <div class="card-icon">📚</div>
                <div class="card-title">Lector de PDF</div>
                <div class="card-description">Extrae texto y datos de documentos PDF.</div>
            </div>
            """, unsafe_allow_html=True
        )
        
with col3:
    with st.container():
        if st.button("PPT Reader", key="ppt_btn"):
            st.switch_page("pages/ppt_reader.py")
        st.markdown(
            f"""
            <div class="card-container">
                <div class="card-icon">📈</div>
                <div class="card-title">Lector de PPT</div>
                <div class="card-description">Analiza el contenido de presentaciones.</div>
            </div>
            """, unsafe_allow_html=True
        )

st.write("") # Spacer

col4, col5, col6 = st.columns(3)

with col4:
    with st.container():
        if st.button("Code Reader", key="code_btn"):
            st.switch_page("pages/code_reader.py")
        st.markdown(
            f"""
            <div class="card-container">
                <div class="card-icon">💻</div>
                <div class="card-title">Lector de Código</div>
                <div class="card-description">Analiza y entiende tus scripts.</div>
            </div>
            """, unsafe_allow_html=True
        )

with col5:
    with st.container():
        if st.button("Image Reader", key="png_btn"):
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