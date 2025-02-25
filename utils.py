import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import streamlit as st
# Definir la función para convertir una columna en tipo datetime
def convert_to_datetime(df, column_name):
    """
    Convierte una columna específica de un DataFrame en formato datetime.

    Parámetros:
    df (pd.DataFrame): DataFrame de pandas.
    column_name (str): Nombre de la columna a convertir.

    Retorna:
    pd.DataFrame: DataFrame con la columna convertida a datetime.
    """
    df[column_name] = pd.to_datetime(df[column_name])
    return df

# Definir la función para graficar una serie de tiempo

def plot_time_series(df, date_col, value_col, title='Serie de Tiempo', x_label='Fecha', y_label='Valor'):
    fig = px.line(df, x=date_col, y=value_col, title=title)
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label, yaxis=dict(range=[0, None], fixedrange=False), 
                      xaxis_showgrid=False, yaxis_showgrid=False)
    return fig



# Definir la función para preparar los datos para el análisis estacional
def preparar_datos_estacionales(df):
    """
    Transforma los datos en un formato adecuado para el análisis estacional.
    
    Args:
        df (pd.DataFrame): DataFrame con columnas 'Fecha' e 'Ingreso'.
    
    Returns:
        pd.DataFrame: DataFrame transformado con columnas 'Año', 'Mes', e 'Ingreso'.
    """
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Año'] = df['Fecha'].dt.year
    df['Mes'] = df['Fecha'].dt.strftime('%b')  # Nombre abreviado del mes
    df['Mes_Numero'] = df['Fecha'].dt.month  # Mes en formato numérico para ordenar
    df = df[['Año', 'Mes', 'Mes_Numero', 'Ingreso']]
    df = df.sort_values(by=['Mes_Numero'])  # Ordenar los meses correctamente
    return df

# Definir la función para graficar el análisis estacional
def graficar_analisis_estacional(df_transformado):
    """
    Genera un gráfico de análisis estacional con los ingresos por mes y líneas por año.
    
    Args:
        df_transformado (pd.DataFrame): DataFrame transformado con columnas 'Año', 'Mes', e 'Ingreso'.
    
    Returns:
        plotly.graph_objects.Figure: Figura de Plotly lista para mostrar en Streamlit.
    """
    fig = px.line(
        df_transformado,
        x='Mes',
        y='Ingreso',
        color='Año',
        markers=True,
        title='Análisis Estacional de Ingresos'
    )
    fig.update_layout(xaxis_title='Mes', yaxis_title='Ingreso', legend_title='Año')
    return fig



def procesar_data_sub_series(df, fecha_col, valor_col):
    """Procesa los datos para la visualización de sub series."""
    df = df.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col])  # Convertir fecha a datetime
    df['Año'] = df[fecha_col].dt.year
    df['Mes'] = df[fecha_col].dt.month
    df['Nombre Mes'] = df[fecha_col].dt.strftime('%B')  # Nombre del mes en texto
    
    # Calcular la media de cada mes en toda la serie temporal
    media_mensual = df.groupby("Mes")[valor_col].mean().to_dict()
    
    return df, media_mensual

def sub_series(df, fecha_col, valor_col):
    """Genera un gráfico de sub series mostrando la evolución de cada mes a lo largo de los años."""
    
    df, media_mensual = procesar_data_sub_series(df, fecha_col, valor_col)
    
    # Crear subgráficos (3 filas, 4 columnas para los 12 meses)
    fig = sp.make_subplots(rows=3, cols=4, subplot_titles=[df[df['Mes'] == i]['Nombre Mes'].values[0] for i in range(1, 13)])

    # Agregar datos a cada subplot
    for i, mes in enumerate(range(1, 13)):  # Del 1 al 12
        df_mes = df[df['Mes'] == mes]
        row = (i // 4) + 1  # Definir fila
        col = (i % 4) + 1  # Definir columna
        
        # Agregar la serie temporal del mes
        fig.add_trace(
            go.Scatter(x=df_mes['Año'], y=df_mes[valor_col], mode='lines+markers', name=f'Mes {mes}'),
            row=row, col=col
        )
        
        # Agregar la media como línea horizontal
        fig.add_trace(
            go.Scatter(
                x=df_mes['Año'], 
                y=[media_mensual[mes]] * len(df_mes['Año']), 
                mode='lines',
                name=f'Media {mes}',
                line=dict(dash='dash', color='red')
            ),
            row=row, col=col
        )

    # Ajustar diseño
    fig.update_layout(height=800, width=1000, title_text="Evolución de la Variable por Mes", showlegend=False)
    
    # Mostrar en Streamlit
    st.plotly_chart(fig)