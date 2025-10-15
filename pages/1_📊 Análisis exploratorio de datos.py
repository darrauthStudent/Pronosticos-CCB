import pandas as pd
import streamlit as st
import numpy as np
from src.etl import convert_to_datetime, ensure_datetime_and_numeric, preparar_datos_estacionales, preparar_datos_subseries, series_indexed
from src.visual_tools import TimeSeriesEDA

# Configuración de página debe ir primero
st.set_page_config(page_title="📊 Cronos - Análisis Exploratorio", layout="wide", initial_sidebar_state="expanded")

def main():

    # Guardia de autenticación
    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        st.error("🔒 Debes iniciar sesión para acceder a esta sección.")
        st.switch_page("Inicio.py")
        st.stop()

    # Verificar que los datos estén cargados
    if not st.session_state.get("data_loaded", False) or st.session_state.get("diccionario_datos") is None:
        st.error("📊 Los datos no están cargados. Regresa a la página principal para cargar los datos.")
        if st.button("🏠 Volver al inicio", use_container_width=True):
            st.switch_page("Inicio.py")
        st.stop()

    # Header principal
    st.title("📊 Análisis Exploratorio de Datos")
    st.caption("Explora patrones, tendencias y características de tus series temporales")
    
    # Obtener datos del diccionario cargado
    diccionario_datos = st.session_state.diccionario_datos
    series_disponibles = list(diccionario_datos.keys())
    
    # Sidebar para selección de serie
    with st.sidebar:
        st.header("🎛️ Panel de Control")
        
        # Selector de serie
        st.subheader("📈 Selección de Serie")
        serie_seleccionada = st.selectbox(
            "Serie temporal a analizar:",
            options=series_disponibles,
            index=0,
            help="Elige la variable que deseas analizar"
        )
        
        # Información de la serie seleccionada
        if serie_seleccionada:
            df_serie = diccionario_datos[serie_seleccionada]
            valores_cero = (df_serie[serie_seleccionada] == 0).sum()
            valores_negativos = (df_serie[serie_seleccionada] < 0).sum()
            
            # Obtener el nombre de la columna de fecha
            fecha_col = None
            for col in df_serie.columns:
                if col.lower() in ['fecha', 'date', 'time', 'timestamp'] or 'fecha' in col.lower():
                    fecha_col = col
                    break
            
            # Si no se encuentra columna de fecha, usar la primera columna
            if fecha_col is None:
                fecha_col = df_serie.columns[0]
            
            # Información de la serie
            st.subheader("📊 Información de la Serie")
            st.info(f"""
            **📈 {serie_seleccionada}**
            
            📊 **Registros:** {len(df_serie):,}
            📅 **Periodo:** {df_serie[fecha_col].min().strftime('%Y-%m-%d')} a {df_serie[fecha_col].max().strftime('%Y-%m-%d')}
            🔢 **Valores cero:** {valores_cero}
            ➖ **Valores negativos:** {valores_negativos}
            """)
            
            # Verificar que se haya seleccionado una serie
            if not serie_seleccionada:
                st.warning("⚠️ Por favor selecciona una serie temporal para analizar.")
                st.stop()
            
            # Obtener DataFrame de la serie seleccionada
            df = diccionario_datos[serie_seleccionada].copy()
            
            # Identificar la columna de fecha en el DataFrame principal
            fecha_col = None
            for col in df.columns:
                if col.lower() in ['fecha', 'date', 'time', 'timestamp'] or 'fecha' in col.lower():
                    fecha_col = col
                    break
            
            # Si no se encuentra, usar la primera columna
            if fecha_col is None:
                fecha_col = df.columns[0]
            
            # Renombrar columnas para compatibilidad con las funciones existentes
            df = df.rename(columns={fecha_col: 'Fecha', serie_seleccionada: 'Ingreso'})
    
    # Aplicar transformaciones - SIMPLIFICADO: Solo serie original
    titulo_transformacion = serie_seleccionada
    unidad_medida = "original"
    
    # Header de análisis
    st.header(f"📈 Análisis de: {titulo_transformacion}")
    
    # Preproceso de fechas para análisis anuales
    df_copy = df.copy()
    df_copy['Fecha'] = pd.to_datetime(df_copy['Fecha'])
    df_copy['Año'] = df_copy['Fecha'].dt.year
    
    # Detectar si la serie es monetaria basándose en palabras clave
    palabras_monetarias = ['ingreso', 'precio', 'costo', 'valor', 'pago', 'venta', 'facturación', 'revenue', 'dinero', 'peso', 'dolar']
    es_monetario = any(keyword in serie_seleccionada.lower() for keyword in palabras_monetarias)
    prefijo_moneda = "$" if es_monetario else ""
    
    # Mostrar estadísticas básicas con métricas nativas de Streamlit
    st.subheader("📊 Estadísticas Principales")
    
    # Primera fila - 4 columnas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total de registros",
            value=f"{len(df):,}"
        )
    
    with col2:
        valor_promedio = df['Ingreso'].mean()
        valor_display = f"{prefijo_moneda}{valor_promedio:,.0f}"
        label_display = "📈 Promedio"
        
        st.metric(
            label=label_display,
            value=valor_display
        )
    
    with col3:
        valor_minimo = df['Ingreso'].min()
        valor_display = f"{prefijo_moneda}{valor_minimo:,.0f}"
        label_display = "📉 Mínimo"
        
        st.metric(
            label=label_display,
            value=valor_display
        )
    
    with col4:
        valor_maximo = df['Ingreso'].max()
        valor_display = f"{prefijo_moneda}{valor_maximo:,.0f}"
        label_display = "📊 Máximo"
        
        st.metric(
            label=label_display,
            value=valor_display
        )
    
    # Segunda fila - 4 columnas para métricas anuales
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        # Promedio anual total
        totales_anuales = df_copy.groupby('Año')['Ingreso'].sum()
        promedio_anual = totales_anuales.mean()
        valor_display = f"{prefijo_moneda}{promedio_anual:,.0f}"
        help_text = "Promedio de los totales anuales"
        
        st.metric(
            label="📅 Promedio anual",
            value=valor_display,
            help=help_text
        )
    
    with col6:
        # Total del año más reciente con datos completos
        años_disponibles = sorted(df_copy['Año'].unique())
        año_completo = None
        
        # Buscar el año más reciente con datos completos (12 meses)
        for año in reversed(años_disponibles):
            datos_año = df_copy[df_copy['Año'] == año]
            if len(datos_año) >= 12:  # Asumiendo datos mensuales
                año_completo = año
                break
        
        if año_completo:
            datos_año_completo = df_copy[df_copy['Año'] == año_completo]
            total_año_completo = datos_año_completo['Ingreso'].sum()
            
            valor_display = f"{prefijo_moneda}{total_año_completo:,.0f}"
            label_display = f"💰 Total {año_completo}"
            help_text = f"Total del año {año_completo} (año más reciente con datos completos)"
        else:
            valor_display = "Sin datos"
            help_text = "No hay años con datos completos"
            label_display = f"💰 Total {2024}"
        
        st.metric(
            label=label_display,
            value=valor_display,
            help=help_text
        )
    
    with col7:
        # Acumulado del último año disponible (año mayor)
        # Obtener el año más reciente con datos
        año_mayor = df_copy['Año'].max()
        
        # Filtrar datos solo para ese año y sumar
        datos_año_mayor = df_copy[df_copy['Año'] == año_mayor]
        total_acumulado = datos_año_mayor['Ingreso'].sum()
        meses_disponibles = len(datos_año_mayor)
        
        valor_display = f"{prefijo_moneda}{total_acumulado:,.0f}"
        label_display = f"📈 Acumulado {año_mayor}"
        help_text = f"Acumulado del año {año_mayor} ({meses_disponibles} meses disponibles)"
        
        st.metric(
            label=label_display,
            value=valor_display,
            help=help_text
        )
    
    with col8:
        # Crecimiento promedio anual
        # Calcular totales anuales
        totales_anuales = df_copy.groupby('Año')['Ingreso'].sum().sort_index()
        
        # Excluir el último año (probablemente incompleto)
        # Solo usar años completos hasta el año anterior al último disponible
        if len(totales_anuales) >= 2:
            # Excluir el último año de los totales
            totales_anuales_completos = totales_anuales.iloc[:-1]
            
            # Calcular crecimiento año a año solo con años completos
            if len(totales_anuales_completos) >= 2:
                crecimientos = []
                años_incluidos = []
                for i in range(1, len(totales_anuales_completos)):
                    año_anterior = totales_anuales_completos.iloc[i-1]
                    año_actual = totales_anuales_completos.iloc[i]
                    if año_anterior > 0:  # Evitar división por cero
                        crecimiento = ((año_actual - año_anterior) / año_anterior) * 100
                        crecimientos.append(crecimiento)
                        años_incluidos.append(f"{totales_anuales_completos.index[i-1]}-{totales_anuales_completos.index[i]}")
                
                if crecimientos:
                    crecimiento_promedio = sum(crecimientos) / len(crecimientos)
                    valor_display = f"{crecimiento_promedio:+.1f}%"
                    
                    # Mostrar rango de años completos en el título
                    año_inicio = totales_anuales_completos.index[0]
                    año_fin = totales_anuales_completos.index[-1]
                    label_display = f"📈 % Crecimiento ({año_inicio}-{año_fin})"
                    
                    help_text = f"Promedio de {len(crecimientos)} períodos (años completos): {', '.join(años_incluidos)}"
                else:
                    valor_display = "N/A"
                    label_display = "📈 % Crecimiento promedio"
                    help_text = "No se puede calcular sin datos válidos"
            else:
                valor_display = "N/A"
                label_display = "📈 % Crecimiento promedio"
                help_text = "Se requieren al menos 2 años completos de datos"
        else:
            valor_display = "N/A"
            label_display = "📈 % Crecimiento promedio"
            help_text = "Se requieren al menos 2 años de datos"
        
        st.metric(
            label=label_display,
            value=valor_display,
            help=help_text
        )

    try:
        # Preproceso de datos
        df = ensure_datetime_and_numeric(df, "Fecha", "Ingreso")
        df_est = preparar_datos_estacionales(df, "Fecha", "Ingreso")
        df_sub, medias = preparar_datos_subseries(df, "Fecha", "Ingreso")
        ser = series_indexed(df, "Fecha", "Ingreso", agg="sum")

        # Crear instancia de análisis
        eda = TimeSeriesEDA(date_col="Fecha", value_col="Ingreso")
        
        # Organizar visualizaciones en tabs para mejor navegación
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Serie Histórica", 
            "🗓️ Análisis Estacional", 
            "📊 Sub-series Mensuales", 
            "🔧 Descomposición"
        ])
        
        with tab1:
            st.subheader("📈 Serie Temporal Histórica")
            st.caption("Visualización completa de la serie temporal a lo largo del tiempo")
            titulo_serie = f"Histórico de {titulo_transformacion}"
            TimeSeriesEDA.render(eda.fig_time_series(df, title=titulo_serie))
            
            # Información adicional sobre la serie
            with st.expander("ℹ️ Información adicional"):
                st.write("""
                **¿Qué puedes observar en esta gráfica?**
                - **Tendencias**: Patrones de crecimiento o decrecimiento a largo plazo
                - **Estacionalidad**: Patrones que se repiten en periodos regulares
                - **Valores atípicos**: Puntos que se desvían significativamente del patrón normal
                - **Volatilidad**: Variabilidad en los datos a lo largo del tiempo
                """)
        
        with tab2:
            st.subheader("🗓️ Análisis Estacional")
            st.caption("Identificación de patrones estacionales en los datos")
            TimeSeriesEDA.render(eda.fig_analisis_estacional(df_est))
            
            with st.expander("ℹ️ Cómo interpretar este análisis"):
                st.write("""
                **Análisis Estacional:**
                - Se puede apreciar más fácilmente patrones mensuales que se mantiene a lo largo de los años, y se hace más fácil identificar cuando hay variaciones en algún año. 
                """)
        
        with tab3:
            st.subheader("📊 Sub-series por Mes")
            st.caption("Análisis detallado de cada mes individual")
            TimeSeriesEDA.render(eda.fig_sub_series(df_sub, medias))
            
            with st.expander("ℹ️ Interpretación de sub-series"):
                st.write("""
                **Sub-series Mensuales:**
                - **Línea horizontal**: Promedio general de cada mes
                - **Puntos**: Valores individuales de cada año para ese mes
                - **Patrones consistentes**: Meses que mantienen comportamiento similar año tras año
                - **Cambios de tendencia**: Meses que muestran evolución a lo largo de los años
                """)
        
        with tab4:
            st.subheader("🔧 Descomposición de la Serie")
            st.caption("Separación de la serie en sus componentes fundamentales")
            
            # Información sobre el método de descomposición usado
            st.info("📋 **Método utilizado**: ADITIVA - Y(t) = Tendencia(t) + Estacional(t) + Residuo(t)")
            
            fig_dec, err = eda.fig_descomposicion(ser, metodo="aditiva", periodo=12)
            if err:
                st.warning(f"⚠️ Error en descomposición: {err}")
                st.info("💡 La descomposición requiere al menos 2 períodos completos de datos.")
            else:
                TimeSeriesEDA.render(fig_dec)
                
                with st.expander("ℹ️ Componentes de la descomposición"):
                    st.write("""
                    **Componentes de la Serie Temporal (Modelo Aditivo):**
                    
                    1. **Serie Observada** (azul): Los datos tal como fueron observados
                    2. **Tendencia** (rojo): Movimiento general a largo plazo, sin fluctuaciones estacionales
                    3. **Estacionalidad** (verde): Patrones que se repiten regularmente, **sumados** a la tendencia
                    4. **Residuo** (naranja): Lo que queda después de remover tendencia y estacionalidad
                    
                    **¿Por qué Descomposición Aditiva?**
                    - **Modelo matemático**: Y(t) = Tendencia(t) + Estacional(t) + Residuo(t)
                    - **Estacionalidad constante**: Los patrones estacionales mantienen amplitud similar a lo largo del tiempo
                    - **Varianza estable**: La variabilidad no aumenta proporcionalmente con el nivel de la serie
                    - **Interpretación directa**: Cada componente se suma algebráicamente
                    
        
                    """)
        
        # Información adicional - SIMPLIFICADA
        # Se removieron las transformaciones para simplificar la experiencia del usuario final
            
    except Exception as e:
        st.error(f"❌ Error procesando los datos: {str(e)}")
        st.info("ℹ️ Verifica que los datos tengan el formato correcto.")

if __name__ == "__main__":
    main()
