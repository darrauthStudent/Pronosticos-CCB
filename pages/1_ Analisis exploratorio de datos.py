import utils as u
import pandas as pd
import streamlit as st

def main():
    st.title("Análisis Exploratorio de Datos")
        
    # leer data 
    data = pd.read_csv('data.csv')
    # convertir la columna 'date' a datetime
    data = u.convert_to_datetime(data, 'Fecha')
 
    # Mostrar el gráfico en Streamlit
    fig1 = u.plot_time_series(data, "Fecha", "Ingreso", title='Histórico de ingresos', x_label='Fecha', y_label='Valor')
    st.plotly_chart(fig1)

    # Preparar datos estacionales y graficar análisis estacional
    df_transformado = u.preparar_datos_estacionales(data)
    fig2 = u.graficar_analisis_estacional(df_transformado)

    # Crear un subconjunto de datos para mostrar en Streamlit
    u.sub_series(data, "Fecha", "Ingreso")

    # Mostrar el segundo gráfico en Streamlit
    st.plotly_chart(fig2)
if __name__ == "__main__":
    main()


