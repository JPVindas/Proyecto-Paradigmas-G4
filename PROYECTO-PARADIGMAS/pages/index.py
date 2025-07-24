import streamlit as st

# Primera fila: 3 botones
left_spacer1, col1, col2, col3, right_spacer1 = st.columns([1, 1, 1, 1, 1])
with col1:
    if st.button(".csv", use_container_width=True):
        st.switch_page("pages/csv_reader.py")
with col2:
    if st.button(".pdf", use_container_width=True):
        st.switch_page("pages/pdf_reader.py")
with col3:
    if st.button(".ppt", use_container_width=True):
        st.switch_page("pages/ppt_reader.py")

# Segunda fila: 2 botones
left_spacer2, col4, col5, right_spacer2 = st.columns([1, 1, 1, 1])
with col4:
    if st.button("code", use_container_width=True):
        st.switch_page("pages/code_reader.py")
with col5:
    if st.button(".png", use_container_width=True):
        st.switch_page("pages/png_reader.py")

# Cierra el contenedor HTML
st.markdown("</div>", unsafe_allow_html=True)
