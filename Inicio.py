import utils as u
import pandas as pd
import streamlit as st

def main():



    # Configurar título y descripción
    st.set_page_config(page_title="Análisis de Series Temporales", layout="wide")

    st.title("📊 Bienvenido a Cronos")
    st.subheader("Explora, descompón y modela tus series temporales")

    st.write("""
    Aquí podrás:
    - 📈 **Explorar la serie de tiempo** y analizar su estacionalidad.
    - 🔍 **Descomponer la serie** en sus componentes fundamentales.
    - 🤖 **Aplicar modelos** y visualizar predicciones.
    - 📊 **Ver un resumen estadístico** de las pruebas realizadas.
    """)

    # Botones de navegación
    st.markdown("## 📌 Navega entre las secciones:")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🔎 Análisis Exploratorio"):
            st.switch_page("pages/1_ Analisis exploratorio de datos.py")

    with col2:
        if st.button("📉 Descomposición de la Serie"):
            st.switch_page("pages/2_descomposicion.py")

    with col3:
        if st.button("📊 Resultados del Modelo"):
            st.switch_page("pages/3_resultados_modelo.py")

    with col4:
        if st.button("📜 Resumen Estadístico"):
            st.switch_page("pages/4_resumen_estadistico.py")

    st.markdown("---")
    st.info("Desarrollado con Streamlit | Visítanos en: [www.miempresa.com](https://www.miempresa.com)")







if __name__ == "__main__":
    main()


