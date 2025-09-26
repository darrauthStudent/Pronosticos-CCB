from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import streamlit as st
from typing import Tuple, Dict, Literal, Optional


#################################################################### FUNCIONES PARA MANEJO DE ARCHIVOS CSV ########################################

def export_dict_to_csv(data_dict, base_path="data/csv"):
    """
    Exporta un diccionario de DataFrames a formato CSV.
    
    Parámetros:
    -----------
    data_dict : dict
        Diccionario con DataFrames a exportar
    base_path : str
        Ruta base donde guardar los archivos .csv
        
    Retorna:
    --------
    list: Lista con los nombres de los archivos exportados
    """
    import os
    
    # Normalizar la ruta base
    base_path = os.path.normpath(base_path)
    
    # Crear directorio para archivos CSV
    os.makedirs(base_path, exist_ok=True)
    exported_files = []
    
    for name, df in data_dict.items():
        # Definir ruta del archivo
        file_name = f"{name}.csv"
        path = os.path.join(base_path, file_name)
        
        # Guardar como CSV con encoding UTF-8
        df.to_csv(path, index=False, encoding='utf-8')
        exported_files.append(file_name)
        print(f"Exportado: {file_name} ({len(df)} filas)")
    
    return exported_files


def load_dict_from_csv(base_path="data/csv"):
    """
    Carga todos los archivos CSV de una carpeta en un diccionario de DataFrames.
    
    Parámetros:
    -----------
    base_path : str
        Ruta base donde están los archivos .csv
        
    Retorna:
    --------
    dict: Diccionario con los DataFrames cargados
    """
    import os
    import pandas as pd
    from pathlib import Path
    
    # Normalizar la ruta base
    base_path = os.path.normpath(base_path)
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"No se encontró la carpeta: {base_path}")
    
    data_dict = {}
    csv_files = Path(base_path).glob("*.csv")
    
    for csv_file in csv_files:
        # El nombre del dataset será el nombre del archivo sin extensión
        name = csv_file.stem
        
        # Cargar DataFrame desde CSV
        df = pd.read_csv(csv_file, encoding='utf-8')
        
        # Convertir la columna Fecha a datetime si existe
        if 'Fecha' in df.columns:
            df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        data_dict[name] = df
    
    if not data_dict:
        raise FileNotFoundError(f"No se encontraron archivos CSV en: {base_path}")
    
    return data_dict


#################################################################### FUNCIONES DE LIMPIEZA Y PREPARACIÓN DE DATOS ########################################

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