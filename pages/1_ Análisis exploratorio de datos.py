import utils as u
import pandas as pd
import streamlit as st

def main():
    st.title("Análisis Exploratorio de Datos")

    # Verificar si los datos ya están cargados en el estado de sesión
    if 'data' not in st.session_state:
        # Cargar los datos y almacenarlos en el estado de sesión
        st.session_state['data'] = pd.read_csv('data.csv')
        # Convertir la columna 'Fecha' a tipo datetime
        st.session_state['data'] = u.convert_to_datetime(st.session_state['data'], 'Fecha')

    # Acceder a los datos desde el estado de sesión
    data = st.session_state['data']

    # Mostrar el gráfico de series temporales
    fig1 = u.plot_time_series(data, "Fecha", "Ingreso", title='Histórico de ingresos', x_label='Fecha', y_label='Valor')
    st.plotly_chart(fig1)

    # Preparar datos estacionales y graficar análisis estacional
    df_transformado = u.preparar_datos_estacionales(data)
    fig2 = u.graficar_analisis_estacional(df_transformado)
    st.plotly_chart(fig2)

    # Crear un subconjunto de datos y mostrarlo
    u.sub_series(data, "Fecha", "Ingreso")

if __name__ == "__main__":
    main()
