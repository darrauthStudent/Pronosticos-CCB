from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import streamlit as st
from typing import Tuple, Dict, Literal, Optional


#################################################################### FUNCIONES DE LIMPIEZA Y PREPARACIÓN DE DATOS ########################################


def export_dict_to_feather(data_dict, base_path="data/feather", manifest_path="data/feather_manifest.json", 
                          compression="lz4", category_threshold=0.5):
    """
    Exporta un diccionario de DataFrames a formato Feather v2 con compresión.
    
    Parámetros:
    -----------
    data_dict : dict
        Diccionario con DataFrames a exportar
    base_path : str
        Ruta base donde guardar los archivos .feather
    manifest_path : str  
        Ruta donde guardar el archivo manifest JSON
    compression : str
        Tipo de compresión ('lz4', 'zstd', 'uncompressed')
    category_threshold : float
        Umbral para convertir columnas a category (0.0-1.0)
        
    Retorna:
    --------
    dict: Manifest con metadatos de los archivos exportados
    """
    import os
    import json
    import pandas as pd
    
    # Normalizar la ruta base para asegurar compatibilidad multiplataforma
    base_path = os.path.normpath(base_path)
    manifest_path = os.path.normpath(manifest_path)
    
    # Crear directorio para archivos Feather
    os.makedirs(base_path, exist_ok=True)
    manifest = []
    
    for name, df in data_dict.items():
        # Crear copia para no modificar el original
        df_optimized = df.copy()
        
        # Optimizaciones de tamaño y rendimiento
        optimizations_applied = []
        for col in df_optimized.select_dtypes("object"):
            # Convierte a category si hay alta repetición
            unique_ratio = df_optimized[col].nunique(dropna=False) / max(len(df_optimized), 1)
            if unique_ratio < category_threshold:
                df_optimized[col] = df_optimized[col].astype("category")
                optimizations_applied.append(f"{col} -> category")
        
        # Definir ruta del archivo usando os.path.join para compatibilidad multiplataforma
        path = os.path.join(base_path, f"{name}.feather")
        
        # Guardar con Feather v2 + compresión
        df_optimized.to_feather(path, compression=compression, version=2)
        
        # Agregar información al manifest
        manifest.append({
            "name": name, 
            "path": path, 
            "rows": len(df_optimized), 
            "cols": list(df_optimized.columns),
            "dtypes": {col: str(dtype) for col, dtype in df_optimized.dtypes.items()},
            "optimizations": optimizations_applied,
            "file_size_mb": round(os.path.getsize(path) / (1024*1024), 2)
        })
    
    # Guardar manifest con metadatos
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    return manifest
    
    # Guardar manifest con metadatos
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    return manifest



# FUNCIÓN PARA CARGAR DE VUELTA LOS DATOS DESDE FEATHER
def load_dict_from_feather(manifest_path="data/feather_manifest.json"):
    """
    Carga un diccionario de DataFrames desde archivos Feather usando el manifest.
    
    Parámetros:
    -----------
    manifest_path : str
        Ruta al archivo manifest JSON
        
    Retorna:
    --------
    dict: Diccionario con los DataFrames cargados
    """
    import json
    import pandas as pd
    import os
    
    # Normalizar la ruta del manifest
    manifest_path = os.path.normpath(manifest_path)
    
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No se encontró el manifest: {manifest_path}")
    
    # Cargar manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    data_dict = {}
    
    for item in manifest:
        # Normalizar la ruta del archivo para asegurar compatibilidad multiplataforma
        file_path = os.path.normpath(item["path"])
        
        # Verificar que el archivo existe antes de intentar cargarlo
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
        
        # Cargar DataFrame desde Feather
        df = pd.read_feather(file_path)
        data_dict[item["name"]] = df
    
    return data_dict







##################################################################################################################################

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


def ensure_datetime_and_numeric(
    df: pd.DataFrame,
    date_col: str = "Fecha",
    value_col: str = "Ingreso",
    drop_na: bool = True,
) -> pd.DataFrame:
    """Convierte fecha a datetime y valor a numérico (si es posible). No usa Streamlit."""
    out = df.copy()

    # Validaciones mínimas
    missing = [c for c in [date_col, value_col] if c not in out.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    # Fecha - Handle Period objects
    if not pd.api.types.is_datetime64_any_dtype(out[date_col]):
        # Check if it's a Period column
        if hasattr(out[date_col].dtype, 'freq') and 'period' in str(out[date_col].dtype).lower():
            # Convert Period to timestamp (datetime) for plotting
            out[date_col] = out[date_col].dt.to_timestamp()
        else:
            out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    # Valor
    if not pd.api.types.is_numeric_dtype(out[value_col]):
        out[value_col] = pd.to_numeric(out[value_col], errors="coerce")

    if drop_na:
        out = out.dropna(subset=[date_col, value_col])

    return out.sort_values(by=date_col).reset_index(drop=True)


def preparar_datos_estacionales(
    df: pd.DataFrame,
    date_col: str = "Fecha",
    value_col: str = "Ingreso",
) -> pd.DataFrame:
    """Crea Año, Mes (abreviado) y Mes_Numero para análisis estacional."""
    out = df.copy()
    out["Año"] = out[date_col].dt.year
    out["Mes"] = out[date_col].dt.strftime("%b")
    out["Mes_Numero"] = out[date_col].dt.month
    cols = ["Año", "Mes", "Mes_Numero", value_col]
    return out[cols].sort_values("Mes_Numero")


def preparar_datos_subseries(
    df: pd.DataFrame,
    date_col: str = "Fecha",
    value_col: str = "Ingreso",
) -> Tuple[pd.DataFrame, Dict[int, float]]:
    """Devuelve DataFrame con Año, Mes (número), Nombre Mes y dict de medias por mes."""
    out = df.copy()
    out["Año"] = out[date_col].dt.year
    out["Mes"] = out[date_col].dt.month
    out["Nombre Mes"] = out[date_col].dt.strftime("%B")
    media_mensual = out.groupby("Mes")[value_col].mean().to_dict()
    return out, media_mensual

def series_indexed(
    df: pd.DataFrame,
    date_col: str = "Fecha",
    value_col: str = "Ingreso",
    agg: Literal["sum", "mean", "first"] = "sum",
) -> pd.Series:
    """Serie con DateTimeIndex para descomposición. Agrega duplicados por fecha según `agg`."""
    tmp = df[[date_col, value_col]].set_index(date_col).sort_index()
    if tmp.index.duplicated().any():
        if agg == "sum":
            tmp = tmp.groupby(level=0).sum()
        elif agg == "mean":
            tmp = tmp.groupby(level=0).mean()
        elif agg == "first":
            tmp = tmp.groupby(level=0).first()
        else:
            raise ValueError("agg debe ser 'sum', 'mean' o 'first'.")
    return tmp[value_col]
