import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.tools import add_constant



def compile_models_output(fitted_values):
    """
    Reorganiza la información de fitted_values en un diccionario estructurado por modelo.
    
    Parameters:
    -----------
    fitted_values : pd.DataFrame
        DataFrame con los valores ajustados de múltiples modelos, incluyendo intervalos de confianza.
        
    Returns:
    --------
    dict
        Diccionario donde cada llave es el nombre del modelo y cada valor es un DataFrame
        con las columnas: unique_id, ds, fitted, lo_95, lo_90, hi_90, hi_95
    """
    result_dict = {}
    
    # Obtener columnas base
    base_cols = ['unique_id', 'ds', 'y']
    model_cols = [col for col in fitted_values.columns if col not in base_cols]
    
    # Identificar modelos únicos
    models = set()
    for col in model_cols:
        model_name = col.split('-')[0]
        models.add(model_name)
    
    # Para cada modelo, extraer sus columnas
    for model in models:
        model_data = fitted_values[base_cols].copy()
        
        # Buscar las columnas del modelo
        fitted_col = model
        lo_95_col = f"{model}-lo-95"
        lo_90_col = f"{model}-lo-90"
        hi_90_col = f"{model}-hi-90"
        hi_95_col = f"{model}-hi-95"
        
        # Verificar que todas las columnas existan
        if all(col in fitted_values.columns for col in [fitted_col, lo_95_col, lo_90_col, hi_90_col, hi_95_col]):
            model_data['fitted'] = fitted_values[fitted_col]
            model_data['lo_95'] = fitted_values[lo_95_col]
            model_data['lo_90'] = fitted_values[lo_90_col]
            model_data['hi_90'] = fitted_values[hi_90_col]
            model_data['hi_95'] = fitted_values[hi_95_col]
            
            result_dict[model] = model_data
    
    return result_dict


def evaluate_models_metrics(models_dict):
    """
    Evalúa múltiples modelos usando métricas RMSE, MAE y MAPE.
    
    Parameters:
    -----------
    models_dict : dict
        Diccionario con los resultados de compile_models_output()
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con las métricas de evaluación para cada modelo
    """

        
    def rmse(y_true, y_pred):
        """Root Mean Square Error"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


    def mae(y_true, y_pred):
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))


    def mape(y_true, y_pred):
        """Mean Absolute Percentage Error"""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    results = []
    
    for model_name, model_data in models_dict.items():
        y_true = model_data['y'].values
        y_pred = model_data['fitted'].values
        
        # Calcular métricas usando nuestras funciones personalizadas
        rmse_val = rmse(y_true, y_pred)
        mae_val = mae(y_true, y_pred)
        mape_val = mape(y_true, y_pred)
        
        results.append({
            'model': model_name,
            'rmse': rmse_val,
            'mae': mae_val,
            'mape': mape_val
        })
    
    # Crear DataFrame y ordenar por RMSE
    metrics_df = pd.DataFrame(results)
    metrics_df = metrics_df.sort_values('rmse').reset_index(drop=True)
    
    return metrics_df



def build_X_future_step(df_y, h=12, freq='M', exog_values=None):
    """
    Construye el DataFrame de variables exógenas para períodos futuros.
    
    Parameters:
    -----------
    df_y : pd.DataFrame
        DataFrame original con columnas 'unique_id', 'ds', 'y' y variables exógenas
    h : int, default=12
        Horizonte de pronóstico (número de períodos futuros)
    freq : str, default='M'
        Frecuencia de las fechas ('M' para mensual, 'D' para diario, etc.)
    exog_values : dict, optional
        Diccionario con valores específicos para cada variable exógena.
        Si no se proporciona, usa el último valor observado de cada variable exógena.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con variables exógenas para períodos futuros
    """
    # Identificar columnas exógenas (todas excepto unique_id, ds, y)
    base_cols = ['unique_id', 'ds', 'y']
    exog_cols = [col for col in df_y.columns if col not in base_cols]
    
    blocks = []
    for uid, g in df_y.groupby('unique_id'):
        last_ds = g['ds'].max()
        future_ds = pd.date_range(last_ds, periods=h+1, freq=freq)[1:]  # próximas h fechas
        
        # Crear DataFrame base para el futuro
        future_data = {
            'unique_id': uid,
            'ds': future_ds,
        }
        
        # Agregar valores para cada variable exógena
        for exog_col in exog_cols:
            if exog_values and exog_col in exog_values:
                # Usar valor específico proporcionado
                future_data[exog_col] = exog_values[exog_col]
            else:
                # Usar el último valor observado
                last_value = g[exog_col].iloc[-1]
                future_data[exog_col] = last_value
        
        blocks.append(pd.DataFrame(future_data))
    
    return pd.concat(blocks, ignore_index=True)


def plot_ets_decomposition(original_data, fitted_model, title="ETS Decomposition", 
                          y_label="Series", figsize=(12, 16)):
    """
    Crea un gráfico de descomposición para modelos ETS con 5 subplots:
    serie original, tendencia, pendiente, estación y residuales.
    
    Parameters:
    -----------
    original_data : array-like
        Los valores originales de la serie temporal
    fitted_model : dict
        El modelo ajustado que contiene 'states' y 'residuals'
    title : str, default="ETS Decomposition"
        Título principal del gráfico
    y_label : str, default="Series"
        Etiqueta para el eje Y del primer subplot
    figsize : tuple, default=(12, 16)
        Tamaño de la figura (ancho, alto)
        
    Returns:
    --------
    fig, axes
        La figura y los ejes de matplotlib para personalización adicional
    """
    # Crear figura con subplots en una columna
    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)
    
    # 1. Serie original
    axes[0].plot(original_data)
    axes[0].set_title(title)
    axes[0].set_ylabel(y_label)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Tendencia (primer componente del estado)
    axes[1].plot(fitted_model["states"][:, 0])
    axes[1].set_ylabel("Tendencia")
    axes[1].grid(True, alpha=0.3)
    
    # 3. Pendiente (segundo componente del estado)
    axes[2].plot(fitted_model["states"][:, 1])
    axes[2].set_ylabel("Pendiente")
    axes[2].grid(True, alpha=0.3)
    
    # 4. Componente estacional (tercer componente del estado)
    axes[3].plot(fitted_model["states"][:, 2])
    axes[3].set_ylabel("Estación")
    axes[3].grid(True, alpha=0.3)
    
    # 5. Residuales
    axes[4].plot(fitted_model["residuals"])
    axes[4].set_ylabel("Residual")
    axes[4].set_xlabel("Tiempo")
    axes[4].grid(True, alpha=0.3)
    axes[4].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig, axes


def plot_residual_analysis(residuals, fitted_values=None, title="Análisis de Residuales", 
                          figsize=(15, 12), bins=15, lags=20):
    """
    Crea un análisis gráfico completo de residuales con 4 subplots:
    histograma con curva normal, Q-Q plot, scatter de valores ajustados vs residuales,
    y autocorrelación de residuales.
    
    Parameters:
    -----------
    residuals : array-like
        Los residuales del modelo
    fitted_values : array-like, optional
        Los valores ajustados del modelo. Si no se proporciona, se usa el índice
    title : str, default="Análisis de Residuales"
        Título principal para el análisis
    figsize : tuple, default=(15, 12)
        Tamaño de la figura (ancho, alto)
    bins : int, default=15
        Número de bins para el histograma
    lags : int, default=20
        Número de lags para el gráfico de autocorrelación
        
    Returns:
    --------
    fig, axes
        La figura y los ejes de matplotlib para personalización adicional
    """
    # Crear figura con subplots 2x2
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Histograma con curva normal
    axes[0,0].hist(residuals, bins=bins, density=True, alpha=0.7, 
                   color='skyblue', edgecolor='black')
    mu, sigma = stats.norm.fit(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0,0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                   label=f'Normal(μ={mu:.4f}, σ={sigma:.4f})')
    axes[0,0].set_title('Histograma de Residuales con Curva Normal')
    axes[0,0].set_xlabel('Residuales')
    axes[0,0].set_ylabel('Densidad')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # 2. Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[0,1])
    axes[0,1].set_title('Q-Q Plot (Normalidad)')
    axes[0,1].grid(True, alpha=0.3)

    # 3. Fitted vs Residuals
    if fitted_values is not None:
        x_vals = fitted_values
        x_label = 'Valores Ajustados'
    else:
        x_vals = range(len(residuals))
        x_label = 'Índice'
    
    axes[1,0].scatter(x_vals, residuals, alpha=0.6, s=30)
    axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
    axes[1,0].set_title('Valores Ajustados vs Residuales')
    axes[1,0].set_xlabel(x_label)
    axes[1,0].set_ylabel('Residuales')
    axes[1,0].grid(True, alpha=0.3)

    # 4. Autocorrelación de residuales
    plot_acf(residuals, lags=lags, ax=axes[1,1], alpha=0.05)
    axes[1,1].set_title('Autocorrelación de Residuales')
    axes[1,1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    return fig, axes


def perform_residual_tests(residuals, title="Pruebas Estadísticas de Residuales", alpha=0.05):
    """
    Realiza un conjunto completo de pruebas estadísticas sobre los residuales de un modelo.
    Incluye pruebas de normalidad, autocorrelación y homocedasticidad.
    
    Parameters:
    -----------
    residuals : array-like
        Los residuales del modelo a analizar
    title : str, default="Pruebas Estadísticas de Residuales"
        Título para el reporte de pruebas
    alpha : float, default=0.05
        Nivel de significancia para las pruebas estadísticas
        
    Returns:
    --------
    dict
        Diccionario con todos los resultados de las pruebas estadísticas
    """
    results = {}
    
    # Convertir a pandas Series para evitar problemas
    residuals_series = pd.Series(residuals)
    
    print(title.upper())
    print("=" * len(title))
    print()
    
    # ========================================
    # 1. PRUEBAS DE NORMALIDAD
    # ========================================
    print("1. PRUEBAS DE NORMALIDAD:")
    print("-" * 30)
    
    # Shapiro-Wilk Test
    try:
        shapiro_stat, shapiro_p = stats.shapiro(residuals_series)
        results['shapiro'] = {'statistic': shapiro_stat, 'p_value': shapiro_p}
        print(f"Shapiro-Wilk Test:")
        print(f"   Estadístico: {shapiro_stat:.6f}")
        print(f"   p-valor: {shapiro_p:.6f}")
        print(f"   Conclusión: {'Residuales NO siguen distribución normal' if shapiro_p < alpha else 'No se rechaza normalidad'}")
        print()
    except Exception as e:
        print(f"Error en Shapiro-Wilk Test: {e}")
        results['shapiro'] = {'error': str(e)}
    
    # Jarque-Bera Test
    try:
        jb_stat, jb_p, skew, kurtosis = jarque_bera(residuals)
        results['jarque_bera'] = {
            'statistic': jb_stat, 
            'p_value': jb_p, 
            'skewness': skew, 
            'kurtosis': kurtosis
        }
        print(f"Jarque-Bera Test:")
        print(f"   Estadístico: {jb_stat:.6f}")
        print(f"   p-valor: {jb_p:.6f}")
        print(f"   Sesgo: {skew:.6f}")
        print(f"   Curtosis: {kurtosis:.6f}")
        print(f"   Conclusión: {'Residuales NO siguen distribución normal' if jb_p < alpha else 'No se rechaza normalidad'}")
        print()
    except Exception as e:
        print(f"Error en Jarque-Bera Test: {e}")
        results['jarque_bera'] = {'error': str(e)}
    
    # Anderson-Darling Test
    try:
        ad_stat, ad_critical, ad_significance = stats.anderson(residuals_series, dist='norm')
        results['anderson_darling'] = {
            'statistic': ad_stat, 
            'critical_values': ad_critical, 
            'significance_levels': ad_significance
        }
        print(f"Anderson-Darling Test:")
        print(f"   Estadístico: {ad_stat:.6f}")
        print(f"   Valor crítico (5%): {ad_critical[2]:.6f}")
        print(f"   Conclusión: {'Residuales NO siguen distribución normal' if ad_stat > ad_critical[2] else 'No se rechaza normalidad'}")
        print()
    except Exception as e:
        print(f"Error en Anderson-Darling Test: {e}")
        results['anderson_darling'] = {'error': str(e)}
    
    print()
    
    # ========================================
    # 2. PRUEBAS DE AUTOCORRELACIÓN
    # ========================================
    print("2. PRUEBAS DE AUTOCORRELACIÓN:")
    print("-" * 30)
    
    # Ljung-Box Test
    try:
        lb_result = acorr_ljungbox(residuals, lags=10, return_df=True)
        lb_stat = lb_result['lb_stat'].iloc[-1]
        lb_p = lb_result['lb_pvalue'].iloc[-1]
        results['ljung_box'] = {'statistic': lb_stat, 'p_value': lb_p}
        
        print(f"Ljung-Box Test (lag 10):")
        print(f"   Estadístico: {lb_stat:.6f}")
        print(f"   p-valor: {lb_p:.6f}")
        print(f"   Conclusión: {'HAY autocorrelación significativa' if lb_p < alpha else 'No hay autocorrelación significativa'}")
        print()
    except Exception as e:
        try:
            # Versión alternativa
            lb_result = acorr_ljungbox(residuals, lags=10, return_df=False)
            if hasattr(lb_result, '__len__') and len(lb_result) == 2:
                lb_stat, lb_p = lb_result
                results['ljung_box'] = {'statistic': lb_stat, 'p_value': lb_p}
                
                print(f"Ljung-Box Test (lag 10):")
                print(f"   Estadístico: {lb_stat:.6f}")
                print(f"   p-valor: {lb_p:.6f}")
                print(f"   Conclusión: {'HAY autocorrelación significativa' if lb_p < alpha else 'No hay autocorrelación significativa'}")
                print()
            else:
                print(f"Error en Ljung-Box Test: formato de resultado no reconocido")
                results['ljung_box'] = {'error': 'Formato de resultado no reconocido'}
        except Exception as e2:
            print(f"Error en Ljung-Box Test: {e2}")
            results['ljung_box'] = {'error': str(e2)}
    
    # Durbin-Watson Test
    try:
        dw_stat = durbin_watson(residuals)
        results['durbin_watson'] = {'statistic': dw_stat}
        
        print(f"Durbin-Watson Test:")
        print(f"   Estadístico: {dw_stat:.6f}")
        if dw_stat < 1.5:
            interpretation = 'Autocorrelación positiva'
        elif dw_stat > 2.5:
            interpretation = 'Autocorrelación negativa'
        else:
            interpretation = 'Sin autocorrelación fuerte'
        print(f"   Interpretación: {interpretation}")
        print()
    except Exception as e:
        print(f"Error en Durbin-Watson Test: {e}")
        results['durbin_watson'] = {'error': str(e)}
    
    print()
    
    # ========================================
    # 3. PRUEBAS DE HOMOCEDASTICIDAD
    # ========================================
    print("3. PRUEBAS DE HOMOCEDASTICIDAD:")
    print("-" * 30)
    
    # Breusch-Pagan Test
    try:
        X = np.arange(len(residuals)).reshape(-1, 1)
        X_with_const = add_constant(X)
        bp_lm, bp_p, bp_fvalue, bp_f_p = het_breuschpagan(residuals, X_with_const)
        results['breusch_pagan'] = {
            'lm_statistic': bp_lm, 
            'lm_p_value': bp_p, 
            'f_statistic': bp_fvalue, 
            'f_p_value': bp_f_p
        }
        
        print(f"Breusch-Pagan Test:")
        print(f"   Estadístico LM: {bp_lm:.6f}")
        print(f"   p-valor: {bp_p:.6f}")
        print(f"   Estadístico F: {bp_fvalue:.6f}")
        print(f"   p-valor F: {bp_f_p:.6f}")
        print(f"   Conclusión: {'HAY heterocedasticidad' if bp_p < alpha else 'No hay heterocedasticidad (homocedasticidad)'}")
        print()
    except Exception as e:
        print(f"Error en Breusch-Pagan Test: {e}")
        results['breusch_pagan'] = {'error': str(e)}
    
    # Levene Test
    try:
        mid = len(residuals) // 2
        group1 = residuals[:mid]
        group2 = residuals[mid:]
        levene_stat, levene_p = stats.levene(group1, group2)
        results['levene'] = {'statistic': levene_stat, 'p_value': levene_p}
        
        print(f"Levene Test (varianzas iguales entre grupos):")
        print(f"   Estadístico: {levene_stat:.6f}")
        print(f"   p-valor: {levene_p:.6f}")
        print(f"   Conclusión: {'Varianzas NO son iguales (heterocedasticidad)' if levene_p < alpha else 'Varianzas son iguales (homocedasticidad)'}")
        print()
    except Exception as e:
        print(f"Error en Levene Test: {e}")
        results['levene'] = {'error': str(e)}
    
    return results


def forecast_with_corrected_dates(sf_model, df, h, level=None, fitted=True, X_df=None, freq='M'):
    """
    Wrapper para sf.forecast que corrige las fechas de pronóstico moviendo una fecha hacia adelante.
    
    Esta función soluciona el problema común donde StatsForecast repite la última fecha observada
    en lugar de generar fechas futuras correctas.
    
    Parameters:
    -----------
    sf_model : StatsForecast
        El modelo StatsForecast ajustado
    df : pd.DataFrame
        El DataFrame con los datos históricos
    h : int
        Horizonte de pronóstico (número de períodos futuros)
    level : list, optional
        Niveles de confianza para intervalos (ej. [90, 95])
    fitted : bool, default=True
        Si incluir valores ajustados
    X_df : pd.DataFrame, optional
        DataFrame con variables exógenas para pronósticos futuros
    freq : str, default='M'
        Frecuencia de las fechas ('M' para mensual, 'D' para diario, etc.)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame de pronósticos con fechas corregidas
    """
    # Generar pronóstico usando StatsForecast
    forecast_result = sf_model.forecast(
        df=df, 
        h=h, 
        level=level, 
        fitted=fitted, 
        X_df=X_df
    )
    
    # Corregir las fechas para cada unique_id
    corrected_forecast = forecast_result.copy()
    
    for unique_id in forecast_result['unique_id'].unique():
        # Filtrar datos para este unique_id
        mask = corrected_forecast['unique_id'] == unique_id
        forecast_subset = corrected_forecast[mask].copy()
        
        # Obtener la última fecha del DataFrame original para este unique_id
        historical_data = df[df['unique_id'] == unique_id]
        last_date = historical_data['ds'].max()
        
        # Generar las fechas correctas (h períodos hacia adelante)
        future_dates = pd.date_range(
            start=last_date, 
            periods=h + 1, 
            freq=freq
        )[1:]  # Excluir la fecha inicial (última observada)
        
        # Actualizar las fechas en el resultado
        corrected_forecast.loc[mask, 'ds'] = future_dates
    
    return corrected_forecast


def create_hybrid_forecast(ets_forecast, sarimax_forecast, target_year=2026, target_month=3,
                          ets_main_col='MMM', sarimax_main_col='auto', hybrid_alias='ESM',
                          show_comparison=True):
    """
    Crea un pronóstico híbrido combinando ETS y SARIMAX para meses específicos.
    
    Parameters:
    -----------
    ets_forecast : pd.DataFrame
        DataFrame con pronósticos ETS
    sarimax_forecast : pd.DataFrame
        DataFrame con pronósticos SARIMAX
    target_year : int, default=2026
        Año objetivo para reemplazar con SARIMAX
    target_month : int, default=3
        Mes objetivo para reemplazar con SARIMAX (3 = marzo)
    ets_main_col : str, default='MMM'
        Nombre de la columna principal en ets_forecast
    sarimax_main_col : str, default='auto'
        Nombre de la columna principal en sarimax_forecast
    hybrid_alias : str, default='ESM'
        Alias para el modelo híbrido en las columnas de salida
    show_comparison : bool, default=True
        Si mostrar comparación de valores para el mes objetivo
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con pronósticos híbridos
    """
    # Preparar tabla base con pronósticos ETS
    hybrid_forecast = ets_forecast.copy()
    
    # Crear mapeo de columnas dinámico
    column_mapping = {}
    for col in ets_forecast.columns:
        if col == 'ds' or col == 'unique_id':
            continue
        if col == ets_main_col:
            column_mapping[col] = hybrid_alias
        elif col.startswith(f"{ets_main_col}-"):
            new_col = col.replace(ets_main_col, hybrid_alias)
            column_mapping[col] = new_col
    
    # Renombrar columnas
    hybrid_forecast = hybrid_forecast.rename(columns=column_mapping)
    
    # Identificar el mes/año objetivo para reemplazar
    target_mask = (hybrid_forecast['ds'].dt.year == target_year) & (hybrid_forecast['ds'].dt.month == target_month)
    
    if target_mask.any():
        # Obtener el valor de SARIMAX para el período objetivo
        target_sarimax = sarimax_forecast[
            (sarimax_forecast['ds'].dt.year == target_year) & 
            (sarimax_forecast['ds'].dt.month == target_month)
        ]
        
        if len(target_sarimax) > 0:
            # Crear mapeo para columnas SARIMAX
            sarimax_mapping = {}
            for col in sarimax_forecast.columns:
                if col == 'ds' or col == 'unique_id':
                    continue
                if col == sarimax_main_col:
                    sarimax_mapping[col] = hybrid_alias
                elif col.startswith(f"{sarimax_main_col}-"):
                    new_col = col.replace(sarimax_main_col, hybrid_alias)
                    sarimax_mapping[col] = new_col
            
            # Reemplazar valores del período objetivo con los del modelo SARIMAX
            for sarimax_col, hybrid_col in sarimax_mapping.items():
                if hybrid_col in hybrid_forecast.columns:
                    hybrid_forecast.loc[target_mask, hybrid_col] = target_sarimax[sarimax_col].iloc[0]
            
        else:
            print(f"No se encontró {target_year}-{target_month:02d} en sarimax_forecast")
    else:
        print(f"No se encontró {target_year}-{target_month:02d} en hybrid_forecast")
    
    # Mostrar comparación si se solicita
    if show_comparison:
        print(f"COMPARACIÓN DE PRONÓSTICOS PARA {target_year}-{target_month:02d}")
        print("-" * 45)
        
        # Filtrar el período objetivo de cada tabla
        target_ets = ets_forecast[target_mask][['ds', ets_main_col]].copy() if target_mask.any() else pd.DataFrame()
        target_sarimax_comp = sarimax_forecast[
            (sarimax_forecast['ds'].dt.year == target_year) & 
            (sarimax_forecast['ds'].dt.month == target_month)
        ][['ds', sarimax_main_col]].copy()
        target_hybrid = hybrid_forecast[target_mask][['ds', hybrid_alias]].copy() if target_mask.any() else pd.DataFrame()
        
        if len(target_ets) > 0:
            print(f"ETS (Original):     {target_ets[ets_main_col].iloc[0]:,.2f}")
        if len(target_sarimax_comp) > 0:
            print(f"SARIMAX:           {target_sarimax_comp[sarimax_main_col].iloc[0]:,.2f}")
        if len(target_hybrid) > 0:
            print(f"{hybrid_alias} (Final):       {target_hybrid[hybrid_alias].iloc[0]:,.2f}")
    
    return hybrid_forecast


def plot_forecast_with_intervals(historical_data, forecast_data, 
                                main_col='ESM', ds_col='ds', y_col='y',
                                title="Pronóstico de Series Temporales",
                                y_label="Valores", figsize=(14, 8),
                                highlight_periods=None):
    """
    Crea un gráfico profesional de serie temporal con pronósticos e intervalos de confianza.
    
    Parameters:
    -----------
    historical_data : pd.DataFrame
        DataFrame con datos históricos (debe tener columnas ds_col y y_col)
    forecast_data : pd.DataFrame
        DataFrame con pronósticos (debe tener main_col y columnas de intervalos)
    main_col : str, default='ESM'
        Nombre de la columna principal del pronóstico
    ds_col : str, default='ds'
        Nombre de la columna de fechas
    y_col : str, default='y'
        Nombre de la columna de valores en datos históricos
    title : str, default="Pronóstico de Series Temporales"
        Título del gráfico
    y_label : str, default="Valores"
        Etiqueta para el eje Y
    figsize : tuple, default=(14, 8)
        Tamaño de la figura
    highlight_periods : list, optional
        Lista de diccionarios con períodos a destacar
        Formato: [{'year': 2026, 'month': 3, 'label': 'Especial', 'color': 'blue'}]
        
    Returns:
    --------
    fig, ax
        Figura y ejes de matplotlib para personalización adicional
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Datos históricos
    ax.plot(historical_data[ds_col], historical_data[y_col], 
            color='black', linewidth=2, label='Datos Históricos', 
            marker='o', markersize=4, alpha=0.8)
    
    # Pronóstico principal
    ax.plot(forecast_data[ds_col], forecast_data[main_col], 
            color='red', linewidth=3, label=f'Pronóstico ({main_col})', 
            marker='s', markersize=6)
    
    # Intervalos de confianza (si existen)
    ic_95_lo = f"{main_col}-lo-95"
    ic_95_hi = f"{main_col}-hi-95"
    ic_90_lo = f"{main_col}-lo-90"
    ic_90_hi = f"{main_col}-hi-90"
    
    if ic_95_lo in forecast_data.columns and ic_95_hi in forecast_data.columns:
        ax.fill_between(forecast_data[ds_col], 
                       forecast_data[ic_95_lo], 
                       forecast_data[ic_95_hi], 
                       alpha=0.2, color='red', label='IC 95%')
    
    if ic_90_lo in forecast_data.columns and ic_90_hi in forecast_data.columns:
        ax.fill_between(forecast_data[ds_col], 
                       forecast_data[ic_90_lo], 
                       forecast_data[ic_90_hi], 
                       alpha=0.3, color='red', label='IC 90%')
    
    # Destacar períodos específicos si se proporcionan
    if highlight_periods:
        for period in highlight_periods:
            year = period.get('year')
            month = period.get('month')
            label = period.get('label', f'{year}-{month:02d}')
            color = period.get('color', 'blue')
            marker = period.get('marker', '*')
            size = period.get('size', 150)
            
            # Filtrar el período específico del pronóstico
            period_mask = (forecast_data[ds_col].dt.year == year) & (forecast_data[ds_col].dt.month == month)
            period_data = forecast_data[period_mask]
            
            if len(period_data) > 0:
                ax.scatter(period_data[ds_col], period_data[main_col], 
                          color=color, s=size, marker=marker, zorder=5, 
                          label=label, edgecolors='white', linewidth=2)
    
    # Configuración del gráfico
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Fecha', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    # Formato de fechas en eje x
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Mejorar el aspecto general
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    plt.tight_layout()
    return fig, ax


def calculate_annual_variation(historical_data, forecast_data, 
                             base_year=2025, target_year=2026,
                             date_col='ds', value_col='y', 
                             forecast_col='ESM', currency='COP',
                             show_results=True):
    """
    Calcula la variación porcentual anual entre dos años consecutivos.
    
    Parameters:
    -----------
    historical_data : pd.DataFrame
        DataFrame con datos históricos
    forecast_data : pd.DataFrame  
        DataFrame con datos de pronóstico
    base_year : int, default=2025
        Año base para la comparación
    target_year : int, default=2026
        Año objetivo para calcular la variación
    date_col : str, default='ds'
        Nombre de la columna de fechas
    value_col : str, default='y'
        Nombre de la columna de valores en datos históricos
    forecast_col : str, default='ESM'
        Nombre de la columna de pronóstico
    currency : str, default='COP'
        Moneda para mostrar en los resultados
    show_results : bool, default=True
        Si mostrar los resultados formateados
        
    Returns:
    --------
    dict
        Diccionario con los resultados del cálculo:
        - 'total_base_year': Total del año base
        - 'total_target_year': Total del año objetivo
        - 'absolute_variation': Variación absoluta
        - 'percentage_variation': Variación porcentual
        - 'base_year': Año base
        - 'target_year': Año objetivo
    """
    import pandas as pd
    
    # Preparar forecast_data con columna year si no existe
    if 'year' not in forecast_data.columns:
        forecast_data = forecast_data.copy()
        forecast_data['year'] = forecast_data[date_col].dt.year
    
    # Obtener datos del año base
    base_data = historical_data[historical_data[date_col].dt.year == base_year]
    total_base = base_data[value_col].sum()
    
    # Obtener pronósticos del año objetivo (primeros 12 meses)
    target_data = forecast_data[forecast_data['year'] == target_year].head(12)
    total_target = target_data[forecast_col].sum()
    
    # Calcular variación
    absolute_variation = total_target - total_base
    percentage_variation = (absolute_variation / total_base) * 100
    
    # Crear resultado
    results = {
        'total_base_year': total_base,
        'total_target_year': total_target,
        'absolute_variation': absolute_variation,
        'percentage_variation': percentage_variation,
        'base_year': base_year,
        'target_year': target_year
    }
    
    # Mostrar resultados formateados si se solicita
    if show_results:
        print(f"📊 VARIACIÓN PORCENTUAL ANUAL ESPERADA PARA {target_year}")
        print("=" * 60)
        print(f"Total {base_year} (Real):        ${total_base:,.0f} millones {currency}")
        print(f"Total {target_year} (Pronóstico):  ${total_target:,.0f} millones {currency}")
        print(f"Variación Absoluta:       ${absolute_variation:+,.0f} millones {currency}")
        print(f"VARIACIÓN PORCENTUAL:     {percentage_variation:+.2f}%")
        print("=" * 60)
    
    return results


def plot_annual_comparison(results_dict, figsize=(10, 6), currency='COP'):
    """
    Crea un gráfico de barras comparativo entre dos años usando los resultados
    de la función calculate_annual_variation.
    
    Parameters:
    -----------
    results_dict : dict
        Diccionario con resultados de calculate_annual_variation
    figsize : tuple, default=(10, 6)
        Tamaño de la figura (ancho, alto)
    currency : str, default='COP'
        Moneda para mostrar en las etiquetas
        
    Returns:
    --------
    fig, ax
        Figura y ejes de matplotlib para personalización adicional
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Datos para el gráfico usando los resultados
    años = [f"{results_dict['base_year']} (Real)", f"{results_dict['target_year']} (Pronóstico)"]
    valores = [results_dict['total_base_year'], results_dict['total_target_year']]
    colores = ['steelblue', 'orange']
    
    # Crear gráfico de barras
    bars = ax.bar(años, valores, color=colores, alpha=0.8, width=0.6)
    
    # Personalizar el gráfico
    ax.set_ylabel(f'Ingresos (Millones {currency})')
    ax.set_title(f'Comparación Anual: Ingresos {results_dict["base_year"]} vs Pronóstico {results_dict["target_year"]}')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores sobre las barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                 f'${height:,.0f}M', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Agregar texto con la variación porcentual
    ax.text(0.5, max(valores) * 0.8, f'Variación: {results_dict["percentage_variation"]:+.2f}%', 
            transform=ax.transData, ha='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    
    # Interpretación simple
    if results_dict['percentage_variation'] > 0:
        interpretacion = f"Se espera un CRECIMIENTO del {results_dict['percentage_variation']:.2f}% en los ingresos para {results_dict['target_year']}"
    else:
        interpretacion = f"Se espera una DISMINUCIÓN del {abs(results_dict['percentage_variation']):.2f}% en los ingresos para {results_dict['target_year']}"
    
    print(f"🎯 CONCLUSIÓN: {interpretacion}")
    
    return fig, ax


def export_data(data, filename, output_dir='data/model_outputs', 
                file_format='csv', add_timestamp=False):
    """
    Exporta un DataFrame a un archivo en el directorio especificado.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame a exportar
    filename : str
        Nombre del archivo (sin extensión)
    output_dir : str, default='data/model_outputs'
        Directorio de salida relativo al directorio raíz del proyecto
    file_format : str, default='csv'
        Formato del archivo ('csv', 'excel', 'parquet')
    add_timestamp : bool, default=False
        Si agregar timestamp al nombre del archivo
        
    Returns:
    --------
    str
        Ruta completa del archivo guardado
    """
    import pandas as pd
    import os
    from datetime import datetime
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Agregar timestamp si se solicita
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp}"
    
    # Determinar extensión y método de guardado
    if file_format.lower() == 'csv':
        file_path = os.path.join(output_dir, f"{filename}.csv")
        data.to_csv(file_path, index=False)
    elif file_format.lower() in ['excel', 'xlsx']:
        file_path = os.path.join(output_dir, f"{filename}.xlsx")
        data.to_excel(file_path, index=False)
    elif file_format.lower() == 'parquet':
        file_path = os.path.join(output_dir, f"{filename}.parquet")
        data.to_parquet(file_path, index=False)
    else:
        raise ValueError(f"Formato no soportado: {file_format}")
    
    # Obtener información del archivo
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"📁 DATOS EXPORTADOS EXITOSAMENTE")
    print("=" * 50)
    print(f"Archivo:     {os.path.basename(file_path)}")
    print(f"Directorio:  {output_dir}")
    print(f"Ruta:        {file_path}")
    print(f"Formato:     {file_format.upper()}")
    print(f"Tamaño:      {file_size_mb:.2f} MB")
    print(f"Filas:       {len(data):,}")
    print(f"Columnas:    {len(data.columns)}")
    print(f"Fecha:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    return file_path