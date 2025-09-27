import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

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

# INTERFAZ PRINCIPAL
def main():
    # Título y descripción
    st.title("📈 Pronósticos de Ingresos por Renovaciones")
    st.markdown("""
    Esta página muestra los pronósticos generados por el modelo híbrido **ETS-SARIMAX** para los ingresos por renovaciones.
    El modelo combina las fortalezas de ambos enfoques para generar predicciones más precisas.
    """)
    
    # Cargar datos
    with st.spinner('Cargando datos...'):
        historical_data = load_historical_data()
        forecast_data = load_forecast_data()
    
    if historical_data is None or forecast_data is None:
        st.error("⚠️ No se pudieron cargar todos los datos necesarios.")
        st.stop()
    
    # Crear filas de métricas (2 filas x 3 columnas)
    # Primera fila
    col1, col2, col3 = st.columns(3)
    
    # Calcular métricas
    last_historical_value = historical_data['Ingresos Renovaciones'].iloc[-1]
    # Corregir: solo sumar pronósticos de 2026, no todos los pronósticos
    forecast_2026_data = forecast_data[forecast_data['Fecha'].dt.year == 2026]
    forecast_2026_total = forecast_2026_data['ESM'].sum()
    historical_2025_total = historical_data[historical_data['Fecha'].dt.year == 2025]['Ingresos Renovaciones'].sum()
    variation_pct = ((forecast_2026_total - historical_2025_total) / historical_2025_total) * 100
    
    # Calcular cierre estimado para 2025 (histórico + pronósticos restantes de 2025)
    forecast_2025_data = forecast_data[forecast_data['Fecha'].dt.year == 2025]
    forecast_2025_remaining = forecast_2025_data['ESM'].sum() if len(forecast_2025_data) > 0 else 0
    estimated_2025_close = historical_2025_total + forecast_2025_remaining
    
    # Calcular promedio histórico de variación anual (2018-2022)
    years_for_avg = [2018, 2019, 2020, 2021, 2022]
    yearly_totals = []
    
    for year in years_for_avg:
        year_data = historical_data[historical_data['Fecha'].dt.year == year]
        if len(year_data) > 0:
            yearly_totals.append(year_data['Ingresos Renovaciones'].sum())
    
    # Calcular variaciones anuales
    variations = []
    for i in range(1, len(yearly_totals)):
        if yearly_totals[i-1] > 0:
            variation = ((yearly_totals[i] - yearly_totals[i-1]) / yearly_totals[i-1]) * 100
            variations.append(variation)
    
    avg_historical_variation = sum(variations) / len(variations) if variations else 0
    
    with col1:
        st.metric(
            label="Último Valor Histórico",
            value=f"${last_historical_value:,.0f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Cierre Estimado 2025",
            value=f"${estimated_2025_close:,.0f}",
            delta=f"+{forecast_2025_remaining:,.0f}" if forecast_2025_remaining > 0 else "Completo"
        )
    
    with col3:
        st.metric(
            label="Pronóstico Total 2026",
            value=f"${forecast_2026_total:,.0f}",
            delta=f"+{(forecast_2026_total - estimated_2025_close):,.0f}"
        )
    
    # Segunda fila
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.metric(
            label="Variación vs 2025",
            value=f"{variation_pct:+.1f}%",
            delta=f"{(forecast_2026_total - estimated_2025_close):+,.0f}"
        )
    
    with col5:
        st.metric(
            label="Promedio Histórico",
            value=f"{avg_historical_variation:+.1f}%",
            delta=f"{min(years_for_avg)}-{max(years_for_avg)}"
        )
    
    with col6:
        st.metric(
            label="Períodos Pronosticados",
            value=f"{len(forecast_data)} meses",
            delta=None
        )
    
    # Gráfico principal
    st.subheader("📊 Visualización de Pronósticos")
    
    fig = create_forecast_chart(historical_data, forecast_data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Información del modelo
    with st.expander("ℹ️ Información del Modelo"):
        st.markdown("""
        **Modelo Híbrido ETS-SARIMAX:**
        
        - **ETS (Error, Trend, Seasonal)**: Captura patrones estacionales y tendencias generales
        - **SARIMAX**: Incorpora variables exógenas (como cambios tarifarios) para mayor precisión
        - **Intervalos de Confianza**: 
          - **90%**: Rango más probable de variación
          - **95%**: Rango más amplio con mayor certeza estadística
        
        **Características del Pronóstico:**
        - Horizonte: 15 meses
        - Frecuencia: Mensual
        - Modelo específico para marzo (efectos de cambios tarifarios)
        """)
    
    # Tabla detallada de pronósticos
    st.subheader("📋 Detalle de Pronósticos")
    
    # Formatear datos para mostrar
    forecast_display = forecast_data.copy()
    forecast_display['Fecha'] = forecast_display['Fecha'].dt.strftime('%Y-%m-%d')
    
    # Renombrar columnas para mejor visualización
    forecast_display = forecast_display.rename(columns={
        'ESM': 'Pronóstico (M COP)',
        'ESM-lo-90': 'IC 90% Inf (M COP)',
        'ESM-hi-90': 'IC 90% Sup (M COP)', 
        'ESM-lo-95': 'IC 95% Inf (M COP)',
        'ESM-hi-95': 'IC 95% Sup (M COP)'
    })
    
    # Formatear números
    numeric_cols = ['Pronóstico (M COP)', 'IC 90% Inf (M COP)', 'IC 90% Sup (M COP)', 
                   'IC 95% Inf (M COP)', 'IC 95% Sup (M COP)']
    
    for col in numeric_cols:
        forecast_display[col] = forecast_display[col].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(forecast_display, use_container_width=True, hide_index=True)
    
    # Botón de descarga después de la tabla
    combined_data = combine_data_for_download(historical_data, forecast_data)
    excel_data = to_excel(combined_data)
    
    st.download_button(
        label="📊 Descargar Resultados de Proyección (Excel)",
        data=excel_data,
        file_name=f"Resultados de proyección de ingresos de Renovación_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Descarga los datos históricos y pronósticos en formato Excel"
    )

if __name__ == "__main__":
    main()