import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os
import glob
import re

# Agregar el directorio src al path para importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from visual_tools import create_forecast_chart, to_excel
from etl import load_historical_data, load_forecast_data, combine_data_for_download

# Configuración de la página
st.set_page_config(
    page_title="Pronósticos - Dashboard Financiero",
    page_icon="📈",
    layout="wide"
)

def normalize_series_name(name):
    """
    Normaliza el nombre de la serie para hacer matching entre datos originales y pronósticos.
    """
    # Convertir a minúsculas y remover espacios extra
    normalized = name.lower().strip()
    
    # Remover la palabra "pronóstico" y "pronostico" al inicio
    normalized = re.sub(r'^(pronóstico|pronostico)\s+', '', normalized)
    
    # Remover extensiones de archivo
    normalized = re.sub(r'\.(csv|xlsx)$', '', normalized)
    
    return normalized

def find_available_series():
    """
    Encuentra todas las series que tienen tanto datos históricos como pronósticos disponibles.
    """
    # Obtener archivos de datos históricos
    historical_files = glob.glob("data/csv/*.csv")
    
    # Obtener archivos de pronósticos
    forecast_files = glob.glob("data/model_outputs/Pronostico*.csv")
    
    # Crear diccionarios de mapeo
    historical_series = {}
    forecast_series = {}
    
    # Procesar archivos históricos
    for file_path in historical_files:
        filename = os.path.basename(file_path)
        name_without_ext = filename.replace('.csv', '')
        normalized_name = normalize_series_name(name_without_ext)
        historical_series[normalized_name] = {
            'display_name': name_without_ext,
            'file_path': file_path,
            'normalized_name': normalized_name
        }
    
    # Procesar archivos de pronósticos
    for file_path in forecast_files:
        filename = os.path.basename(file_path)
        name_without_ext = filename.replace('.csv', '')
        normalized_name = normalize_series_name(name_without_ext)
        forecast_series[normalized_name] = {
            'display_name': name_without_ext,
            'file_path': file_path,
            'normalized_name': normalized_name
        }
    
    # Encontrar series que tienen tanto datos históricos como pronósticos
    available_series = {}
    for normalized_name in historical_series:
        if normalized_name in forecast_series:
            available_series[normalized_name] = {
                'display_name': historical_series[normalized_name]['display_name'],
                'historical_file': historical_series[normalized_name]['file_path'],
                'forecast_file': forecast_series[normalized_name]['file_path']
            }
    
    return available_series

def load_series_data(historical_file, forecast_file):
    """
    Carga los datos históricos y de pronósticos para una serie específica.
    """
    try:
        # Cargar datos históricos
        historical_data = pd.read_csv(historical_file, parse_dates=['Fecha'])
        
        # Cargar datos de pronósticos
        forecast_data = pd.read_csv(forecast_file, parse_dates=['Fecha'])
        
        return historical_data, forecast_data
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        return None, None

def create_dynamic_forecast_chart(historical_data, forecast_data, value_column, forecast_main_column, series_name):
    """
    Crea un gráfico dinámico de pronósticos que se adapta a cualquier serie temporal.
    """
    fig = go.Figure()
    
    # Línea histórica
    fig.add_trace(go.Scatter(
        x=historical_data['Fecha'],
        y=historical_data[value_column],
        mode='lines+markers',
        name='Datos Históricos',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4)
    ))
    
    # Línea de pronóstico principal
    fig.add_trace(go.Scatter(
        x=forecast_data['Fecha'],
        y=forecast_data[forecast_main_column],
        mode='lines+markers',
        name='Pronóstico',
        line=dict(color='#ff7f0e', width=2, dash='dot'),
        marker=dict(size=4)
    ))
    
    # Intervalos de confianza (si están disponibles)
    # Buscar columnas de intervalos de confianza
    ic_90_lo = None
    ic_90_hi = None
    ic_95_lo = None
    ic_95_hi = None
    
    for col in forecast_data.columns:
        if '-lo-90' in col or 'lo_90' in col:
            ic_90_lo = col
        elif '-hi-90' in col or 'hi_90' in col:
            ic_90_hi = col
        elif '-lo-95' in col or 'lo_95' in col:
            ic_95_lo = col
        elif '-hi-95' in col or 'hi_95' in col:
            ic_95_hi = col
    
    # Agregar intervalos de confianza del 95% si están disponibles
    if ic_95_lo and ic_95_hi:
        fig.add_trace(go.Scatter(
            x=list(forecast_data['Fecha']) + list(forecast_data['Fecha'][::-1]),
            y=list(forecast_data[ic_95_hi]) + list(forecast_data[ic_95_lo][::-1]),
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='IC 95%',
            showlegend=True,
            hoverinfo='skip'  # No mostrar información al pasar el mouse
        ))
    
    # Agregar intervalos de confianza del 90% si están disponibles
    if ic_90_lo and ic_90_hi:
        fig.add_trace(go.Scatter(
            x=list(forecast_data['Fecha']) + list(forecast_data['Fecha'][::-1]),
            y=list(forecast_data[ic_90_hi]) + list(forecast_data[ic_90_lo][::-1]),
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='IC 90%',
            showlegend=True,
            hoverinfo='skip'  # No mostrar información al pasar el mouse
        ))
    
    # Configuración del gráfico
    fig.update_layout(
        title=f'Pronóstico de {series_name}',
        xaxis_title='Fecha',
        yaxis_title='Valor (COP)',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        # Formatear el eje Y para mostrar números con comas
        yaxis=dict(tickformat='.0f')
    )
    
    return fig

# INTERFAZ PRINCIPAL
def main():
    # Título y descripción
    st.title("📈 Pronósticos de Series Temporales")
    st.markdown("""
    Esta página muestra los pronósticos generados por modelos de series temporales.
    Selecciona la serie que deseas analizar en el panel lateral.
    """)
    
    # Sidebar para selección de serie
    with st.sidebar:
        st.header("🎛️ Panel de Control")
        
        # Encontrar series disponibles
        available_series = find_available_series()
        
        if not available_series:
            st.error("⚠️ No se encontraron series con pronósticos disponibles.")
            st.stop()
        
        # Crear selector de series
        st.subheader("📈 Selección de Serie")
        series_names = list(available_series.keys())
        series_display_names = [available_series[key]['display_name'] for key in series_names]
        
        selected_display_name = st.selectbox(
            "Serie temporal a analizar:",
            options=series_display_names,
            index=0,
            help="Elige la serie que deseas analizar"
        )
        
        # Encontrar la serie seleccionada
        selected_series_key = None
        for key, data in available_series.items():
            if data['display_name'] == selected_display_name:
                selected_series_key = key
                selected_series_data = data
                break
        
        # Información de la serie seleccionada
        st.subheader("📊 Información de la Serie")
        st.info(f"""
        **📈 {selected_display_name}**
        
        🔗 **Archivos vinculados:**
        - Datos históricos disponibles ✅
        - Pronósticos disponibles ✅
        """)
    
    # Cargar datos de la serie seleccionada
    with st.spinner(f'Cargando datos de {selected_display_name}...'):
        historical_data, forecast_data = load_series_data(
            selected_series_data['historical_file'],
            selected_series_data['forecast_file']
        )
    
    if historical_data is None or forecast_data is None:
        st.error("⚠️ No se pudieron cargar todos los datos necesarios.")
        st.stop()
    
    # Detectar la columna de valores en los datos históricos (no es siempre igual)
    value_columns = [col for col in historical_data.columns if col != 'Fecha']
    if len(value_columns) == 0:
        st.error("⚠️ No se encontró columna de valores en los datos históricos.")
        st.stop()
    
    value_column = value_columns[0]  # Tomar la primera columna de valores
    
    # Detectar la columna de pronóstico principal (buscar la que no tenga sufijos de intervalos)
    forecast_columns = [col for col in forecast_data.columns if col not in ['Fecha', 'unique_id'] and not any(suffix in col.lower() for suffix in ['-lo-', '-hi-', 'lo_', 'hi_'])]
    if len(forecast_columns) == 0:
        st.error("⚠️ No se encontró columna de pronóstico principal.")
        st.stop()
    
    forecast_main_column = forecast_columns[0]  # Tomar la primera columna de pronóstico principal
    
    # Crear filas de métricas redistribuidas
    # Primera fila - 3 columnas
    col1, col2, col3 = st.columns(3)
    
    # Calcular métricas dinámicamente
    # Obtener el año más reciente con datos históricos
    max_historical_year = historical_data['Fecha'].dt.year.max()
    
    # Corregir: solo sumar pronósticos del año siguiente al último histórico
    next_year = max_historical_year + 1
    forecast_next_year_data = forecast_data[forecast_data['Fecha'].dt.year == next_year]
    forecast_next_year_total = forecast_next_year_data[forecast_main_column].sum() if len(forecast_next_year_data) > 0 else 0
    
    # Total del último año histórico
    historical_last_year_data = historical_data[historical_data['Fecha'].dt.year == max_historical_year]
    historical_last_year_total = historical_last_year_data[value_column].sum() if len(historical_last_year_data) > 0 else 0
    
    # Calcular cierre estimado para el año actual (histórico + pronósticos restantes del año actual)
    current_year_forecast_data = forecast_data[forecast_data['Fecha'].dt.year == max_historical_year]
    current_year_forecast_remaining = current_year_forecast_data[forecast_main_column].sum() if len(current_year_forecast_data) > 0 else 0
    estimated_current_year_close = historical_last_year_total + current_year_forecast_remaining
    
    # Calcular variación porcentual correctamente (2026 vs cierre estimado completo 2025)
    variation_pct = 0
    if estimated_current_year_close > 0 and forecast_next_year_total > 0:
        variation_pct = ((forecast_next_year_total - estimated_current_year_close) / estimated_current_year_close) * 100
    
    # Calcular promedio histórico de variación anual (últimos 5 años disponibles)
    available_years = sorted(historical_data['Fecha'].dt.year.unique())
    years_for_avg = available_years[-5:] if len(available_years) >= 5 else available_years[:-1]  # Excluir el último año si está incompleto
    
    yearly_totals = []
    for year in years_for_avg:
        year_data = historical_data[historical_data['Fecha'].dt.year == year]
        if len(year_data) >= 10:  # Solo considerar años con datos suficientes (al menos 10 meses)
            yearly_totals.append(year_data[value_column].sum())
    
    # Calcular variaciones anuales
    variations = []
    for i in range(1, len(yearly_totals)):
        if yearly_totals[i-1] > 0:
            variation = ((yearly_totals[i] - yearly_totals[i-1]) / yearly_totals[i-1]) * 100
            variations.append(variation)
    
    avg_historical_variation = sum(variations) / len(variations) if variations else 0
    
    with col1:
        st.metric(
            label=f"Cierre Estimado {max_historical_year}",
            value=f"${estimated_current_year_close:,.0f}",
            delta=f"+{current_year_forecast_remaining:,.0f}" if current_year_forecast_remaining > 0 else "Completo"
        )
    
    with col2:
        st.metric(
            label=f"Pronóstico Total {next_year}",
            value=f"${forecast_next_year_total:,.0f}",
            delta=f"+{(forecast_next_year_total - estimated_current_year_close):,.0f}" if forecast_next_year_total > 0 else "Sin datos"
        )
    
    with col3:
        st.metric(
            label="Períodos Pronosticados",
            value=f"{len(forecast_data)} meses",
            delta=None
        )
    
    # Segunda fila - 2 columnas alineadas con posiciones 1 y 2 de arriba
    col4, col5, _ = st.columns(3)
    
    with col4:
        st.metric(
            label=f"Variación vs {max_historical_year}",
            value=f"{variation_pct:+.1f}%",
            delta=f"{(forecast_next_year_total - estimated_current_year_close):+,.0f}" if forecast_next_year_total > 0 and estimated_current_year_close > 0 else "N/A"
        )
    
    with col5:
        year_range_display = f"{years_for_avg[0]}-{years_for_avg[-1]}" if len(years_for_avg) >= 2 else "Insuficientes datos"
        st.metric(
            label=f"Promedio Histórico ({year_range_display})",
            value=f"{avg_historical_variation:+.1f}%" if variations else "N/A",
            delta=None
        )
    
    # Gráfico principal
    st.subheader("📊 Visualización de Pronósticos")
    
    # Crear gráfico dinámico
    fig = create_dynamic_forecast_chart(historical_data, forecast_data, value_column, forecast_main_column, selected_display_name)
    st.plotly_chart(fig, use_container_width=True)
    
    # Información del modelo
    with st.expander("ℹ️ Información del Modelo"):
        st.markdown(f"""
        **Serie:** {selected_display_name}
        
        **Características del Pronóstico:**
        - Horizonte: {len(forecast_data)} meses
        - Frecuencia: Mensual
        - Período histórico: {historical_data['Fecha'].min().strftime('%Y-%m')} a {historical_data['Fecha'].max().strftime('%Y-%m')}
        - Período pronosticado: {forecast_data['Fecha'].min().strftime('%Y-%m')} a {forecast_data['Fecha'].max().strftime('%Y-%m')}
        
        **Intervalos de Confianza:**
        - **90%**: Rango más probable de variación
        - **95%**: Rango más amplio con mayor certeza estadística
        """)
    
    # Tabla detallada de pronósticos
    st.subheader("📋 Detalle de Pronósticos")
    
    # Formatear datos para mostrar
    forecast_display = forecast_data.copy()
    forecast_display['Fecha'] = forecast_display['Fecha'].dt.strftime('%Y-%m-%d')
    
    # Renombrar columnas dinámicamente para mejor visualización
    column_mapping = {'Fecha': 'Fecha'}
    
    # Identificar y renombrar la columna principal de pronóstico
    column_mapping[forecast_main_column] = 'Pronóstico (COP)'
    
    # Identificar y renombrar columnas de intervalos de confianza
    for col in forecast_data.columns:
        if col not in ['Fecha', 'unique_id', forecast_main_column]:
            if '-lo-90' in col or 'lo_90' in col:
                column_mapping[col] = 'IC 90% Inf (COP)'
            elif '-hi-90' in col or 'hi_90' in col:
                column_mapping[col] = 'IC 90% Sup (COP)'
            elif '-lo-95' in col or 'lo_95' in col:
                column_mapping[col] = 'IC 95% Inf (COP)'
            elif '-hi-95' in col or 'hi_95' in col:
                column_mapping[col] = 'IC 95% Sup (COP)'
    
    # Aplicar renombrado
    forecast_display = forecast_display.rename(columns=column_mapping)
    
    # Formatear números
    numeric_cols = [col for col in forecast_display.columns if 'COP' in col]
    
    for col in numeric_cols:
        if col in forecast_display.columns:
            forecast_display[col] = forecast_display[col].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(forecast_display, use_container_width=True, hide_index=True)
    
    # Botón de descarga después de la tabla
    combined_data = combine_dynamic_data_for_download(historical_data, forecast_data, value_column, forecast_main_column)
    excel_data = to_excel(combined_data)
    
    # Normalizar el nombre para el archivo
    file_safe_name = selected_display_name.replace(" ", "_").replace("ñ", "n")
    
    st.download_button(
        label=f"📊 Descargar Resultados de {selected_display_name} (Excel)",
        data=excel_data,
        file_name=f"Resultados_proyeccion_{file_safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Descarga los datos históricos y pronósticos en formato Excel"
    )

def combine_dynamic_data_for_download(historical_data, forecast_data, value_column, forecast_main_column):
    """
    Combina datos históricos y de pronósticos para descarga, adaptándose a cualquier serie.
    """
    # Preparar datos históricos
    historical_prep = historical_data.copy()
    historical_prep['Tipo'] = 'Histórico'
    historical_prep = historical_prep.rename(columns={value_column: 'Valor'})
    
    # Preparar datos de pronósticos
    forecast_prep = forecast_data.copy()
    forecast_prep['Tipo'] = 'Pronóstico'
    forecast_prep = forecast_prep.rename(columns={forecast_main_column: 'Valor'})
    
    # Combinar solo las columnas necesarias
    columns_to_keep = ['Fecha', 'Valor', 'Tipo']
    
    # Agregar columnas de intervalos de confianza si están disponibles
    for col in forecast_data.columns:
        if any(pattern in col for pattern in ['-lo-90', '-hi-90', '-lo-95', '-hi-95', 'lo_90', 'hi_90', 'lo_95', 'hi_95']):
            if col in forecast_prep.columns:
                columns_to_keep.append(col)
    
    # Filtrar columnas existentes
    historical_columns = [col for col in columns_to_keep if col in historical_prep.columns]
    forecast_columns = [col for col in columns_to_keep if col in forecast_prep.columns]
    
    historical_prep = historical_prep[historical_columns]
    forecast_prep = forecast_prep[forecast_columns]
    
    # Combinar datasets
    combined_data = pd.concat([historical_prep, forecast_prep], ignore_index=True, sort=False)
    combined_data = combined_data.sort_values('Fecha')
    
    return combined_data

if __name__ == "__main__":
    main()