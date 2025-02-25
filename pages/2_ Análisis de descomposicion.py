import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
import utils as u


# Implementación en Streamlit
st.title('Descomposición Estacional de Series de Tiempo')

# Verificar si los datos ya están en el estado de sesión
if 'data' not in st.session_state:
    archivo = st.file_uploader("Sube un archivo CSV con datos de series de tiempo", type=['csv'])
    if archivo is not None:
        df = pd.read_csv(archivo, parse_dates=[0], index_col=0)
        st.session_state['data'] = df
    else:
        st.warning("Por favor, sube un archivo CSV para continuar.")
        st.stop()
else:
    df = st.session_state['data']

columna = st.selectbox("Selecciona la columna de la serie de tiempo", df.columns)

# Definir opciones de frecuencia
opciones_frecuencia = {
    'Diaria': 1,
    'Semanal': 7,
    'Mensual': 30,
    'Anual': 365
}

# Crear selectbox para la frecuencia
frecuencia_seleccionada = st.selectbox("Selecciona la frecuencia de la estacionalidad", list(opciones_frecuencia.keys()))

# Obtener el periodo correspondiente a la frecuencia seleccionada
periodo = opciones_frecuencia[frecuencia_seleccionada]

metodo = st.selectbox("Selecciona el método de descomposición", ['aditiva', 'multiplicativa', 'stl', 'mstl'])

if st.button("Aplicar Descomposición"):
    fig = u.descomposicion_estacional(df, columna, periodo, metodo)
    if fig:
        st.plotly_chart(fig)