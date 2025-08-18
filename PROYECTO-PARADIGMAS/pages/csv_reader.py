# leer_csv.py
import streamlit as st
import pandas as pd
import numpy as np

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
    plot_clusters_pca
)

st.title("📂 Lectura y análisis de CSV")

# --------------------
# Subir archivo CSV
# --------------------
archivo = st.file_uploader("Sube tu CSV", type=["csv"])

if archivo:
    try:
        with st.spinner("Cargando CSV..."):
            df = cargar_dataframe(archivo)
            st.success(f"CSV cargado: {len(df)} filas × {len(df.columns)} columnas")
            st.dataframe(df.head(10))

        # Detectar tipos
        tipo_df = detectar_tipos(df)
        st.subheader("📊 Tipos de datos detectados")
        st.dataframe(tipo_df, use_container_width=True)

        # Columnas numéricas y categóricas
        num_cols = tipo_df[tipo_df["Tipo"] == "Numérica"]["Variable"].tolist()
        cat_cols = tipo_df[tipo_df["Tipo"] == "Categórica"]["Variable"].tolist()

        # Estadísticas descriptivas
        st.subheader("📈 Estadísticas descriptivas (numéricas)")
        st.dataframe(df[num_cols].describe(), use_container_width=True)

        # Visualizaciones
        st.subheader("🔍 Visualizaciones automáticas")
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

        # Mostrar variables seleccionadas
        st.markdown("**Variables usadas en gráficas:**")
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

        # Histogramas individuales
        for var in selected_nums:
            fig_h = generar_histograma_claro(df, var)
            if fig_h:
                st.plotly_chart(fig_h, use_container_width=True)

        # Boxplots comparativos
        fig_box = generar_boxplot_claro(df, selected_nums, max_display=6)
        if fig_box:
            st.plotly_chart(fig_box, use_container_width=True)

        # Correlación
        fig_corr = generar_correlacion_clara(df, num_cols)
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True)

        # Barras categóricas
        if selected_cat:
            fig_bar = generar_barras_claras(df, selected_cat, top_n=20)
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)

        # Violin + Box
        if selected_nums:
            fig_vb = generar_violin_y_box(df, selected_nums[0])
            if fig_vb:
                st.plotly_chart(fig_vb, use_container_width=True)

        # Outliers
        st.subheader("⚠️ Detección de outliers")
        if selected_nums:
            contamination = st.slider("Sensibilidad outliers", 0.01, 0.2, 0.05, 0.01, key="contamination_csv")
            outliers = detectar_outliers(df, selected_nums, contamination)
            if not outliers.empty:
                frac = len(outliers) / max(1, len(df))
                st.info(f"Outliers detectados: {len(outliers)} — {frac:.2%} del dataset")
                st.dataframe(outliers.head(200))
            else:
                st.success("No se detectaron outliers.")

        # Clustering
        st.subheader("🧩 Clustering (KMeans / MiniBatch)")
        if num_cols and len(num_cols) >= 2:
            n_clusters = st.slider("Número de clusters", 2, 12, 3, key="nclusters_csv")
            df_clusters_vis, model_info = clustering_kmeans(df, num_cols, n_clusters=n_clusters, minibatch=True)
            if df_clusters_vis is not None:
                st.info(f"Muestra de {len(df_clusters_vis)} observaciones para visualizar clusters")
                st.dataframe(df_clusters_vis.head(100))
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
    st.info("Sube un archivo CSV para comenzar el análisis.")
