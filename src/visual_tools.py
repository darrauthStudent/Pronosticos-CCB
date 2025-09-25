# utils_eda.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
from typing import Dict, Tuple, Optional


class TimeSeriesEDA:
    """
    Solo diseña figuras (Plotly) y ofrece un renderizador genérico para Streamlit.
    El preprocesamiento/derivación de datasets debe hacerse en utils_prep.py
    """

    def __init__(self, date_col: str = "Fecha", value_col: str = "Ingreso") -> None:
        self.date_col = date_col
        self.value_col = value_col

    # -------------------- Figuras (reciben datos ya preparados) -------------------- #
    def fig_time_series(self, df: pd.DataFrame, title: str = "Serie de Tiempo"):
        return px.line(df, x=self.date_col, y=self.value_col, title=title)

    def fig_analisis_estacional(self, df_estacional: pd.DataFrame, title: str = "Análisis Estacional"):
        """
        Espera columnas: 'Año', 'Mes', 'Mes_Numero' y `value_col`.
        """
        fig = px.line(
            df_estacional,
            x="Mes",
            y=self.value_col,
            color="Año",
            markers=True,
            title=title,
        )
        fig.update_layout(xaxis_title="Mes", yaxis_title=self.value_col, legend_title="Año")
        return fig

    def fig_sub_series(
        self,
        df_sub: pd.DataFrame,
        media_mensual: Dict[int, float],
        title: str = "Evolución por Mes",
        height: int = 800,
        width: int = 1000,
    ):
        """
        Espera df_sub con columnas: 'Año', 'Mes' (número 1..12), 'Nombre Mes' y `value_col`.
        """
        fig = make_subplots(
            rows=3,
            cols=4,
            subplot_titles=[
                df_sub[df_sub["Mes"] == i]["Nombre Mes"].values[0] if (df_sub["Mes"] == i).any() else f"Mes {i}"
                for i in range(1, 13)
            ],
        )

        for i, mes in enumerate(range(1, 13)):
            df_mes = df_sub[df_sub["Mes"] == mes]
            if df_mes.empty:
                continue
            row, col = i // 4 + 1, i % 4 + 1
            fig.add_scatter(x=df_mes["Año"], y=df_mes[self.value_col], mode="lines+markers", row=row, col=col)
            fig.add_scatter(
                x=df_mes["Año"],
                y=[media_mensual.get(mes, None)] * len(df_mes["Año"]),
                mode="lines",
                line=dict(dash="dash"),
                row=row,
                col=col,
            )

        fig.update_layout(title=title, height=height, width=width, showlegend=False)
        return fig

    def fig_descomposicion(
        self,
        series: pd.Series,
        metodo: str,
        periodo: int,
        title: str = "Descomposición Estacional",
    ) -> Tuple[Optional[go.Figure], Optional[str]]:
        """
        `series` debe ser pd.Series con DateTimeIndex (ya construida en utils_prep.series_indexed).
        metodo: 'aditiva' | 'multiplicativa' | 'stl' | 'mstl'
        """
        if len(series) < 2 * periodo:
            return None, f"Se requieren al menos {2 * periodo} observaciones (actual={len(series)})."

        # Mapear métodos a nombres descriptivos
        metodo_nombres = {
            'aditiva': 'ADITIVA',
            'multiplicativa': 'MULTIPLICATIVA', 
            'stl': 'STL (Seasonal and Trend decomposition using Loess)',
            'mstl': 'MSTL (Multiple Seasonal-Trend decomposition using Loess)'
        }
        
        metodo_display = metodo_nombres.get(metodo, metodo.upper())

        if metodo in ("aditiva", "multiplicativa"):
            res = seasonal_decompose(series, model=metodo, period=periodo)
            observed, trend, seasonal, resid = res.observed, res.trend, res.seasonal, res.resid
        elif metodo == "stl":
            res = STL(series, period=periodo).fit()
            observed, trend, seasonal, resid = res.observed, res.trend, res.seasonal, res.resid
        elif metodo == "mstl":
            res = MSTL(series, periods=[periodo]).fit()
            observed, trend, seasonal, resid = res.observed, res.trend, res.seasonal, None
        else:
            return None, "Método no reconocido."

        rows = 4 if resid is not None else 3
        
        # Títulos descriptivos para cada componente
        subplot_titles = ["Serie Observada", "Tendencia", "Estacionalidad"]
        if resid is not None:
            subplot_titles.append("Residuo")
            
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            subplot_titles=subplot_titles,
        )
        
        # Agregar trazas con nombres descriptivos en la leyenda
        fig.add_scatter(
            x=observed.index, 
            y=observed, 
            mode="lines", 
            row=1, 
            col=1,
            name="Serie Observada",
            line=dict(color='blue')
        )
        fig.add_scatter(
            x=trend.index, 
            y=trend, 
            mode="lines", 
            row=2, 
            col=1,
            name="Tendencia",
            line=dict(color='red')
        )
        fig.add_scatter(
            x=seasonal.index, 
            y=seasonal, 
            mode="lines", 
            row=3, 
            col=1,
            name="Estacionalidad",
            line=dict(color='green')
        )
        if resid is not None:
            fig.add_scatter(
                x=resid.index, 
                y=resid, 
                mode="lines", 
                row=4, 
                col=1,
                name="Residuo",
                line=dict(color='orange')
            )
            
        # Título mejorado que incluye el tipo de descomposición
        titulo_completo = f"{title} - Método: {metodo_display}"
        
        fig.update_layout(
            title=titulo_completo, 
            height=800, 
            width=900,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        fig.update_xaxes(title_text="Fecha")
        fig.update_yaxes(title_text=self.value_col)
        return fig, None

    # -------------------- Único render genérico -------------------- #
    @staticmethod
    def render(fig, **kwargs):
        """Muestra cualquier figura Plotly en Streamlit."""
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, **kwargs)

