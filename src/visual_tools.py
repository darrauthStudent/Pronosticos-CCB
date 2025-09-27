# utils_eda.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
from typing import Dict, Tuple, Optional
import io


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


def create_forecast_chart(historical_df, forecast_df):
    """Crea un gráfico interactivo con datos históricos y pronósticos"""
    
    fig = go.Figure()
    
    # Datos históricos
    if historical_df is not None:
        fig.add_trace(go.Scatter(
            x=historical_df['Fecha'],
            y=historical_df['Ingresos Renovaciones'],
            mode='lines+markers',
            name='Datos Históricos',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Fecha:</b> %{x}<br><b>Ingresos:</b> $%{y:,.0f}M COP<extra></extra>'
        ))
    
    # Pronósticos
    if forecast_df is not None:
        # Línea principal del pronóstico
        fig.add_trace(go.Scatter(
            x=forecast_df['Fecha'],
            y=forecast_df['ESM'],
            mode='lines+markers',
            name='Pronóstico Híbrido (ESM)',
            line=dict(color='#F18F01', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond'),
            hovertemplate='<b>Fecha:</b> %{x}<br><b>Pronóstico:</b> $%{y:,.0f}M COP<extra></extra>'
        ))
        
        # Intervalo de confianza 95%
        fig.add_trace(go.Scatter(
            x=forecast_df['Fecha'],
            y=forecast_df['ESM-hi-95'],
            mode='lines',
            name='IC 95% Superior',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['Fecha'],
            y=forecast_df['ESM-lo-95'],
            mode='lines',
            name='Intervalo de Confianza 95%',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(241, 143, 1, 0.2)',
            hovertemplate='<b>Fecha:</b> %{x}<br><b>IC 95%:</b> $%{y:,.0f}M COP<extra></extra>'
        ))
        
        # Intervalo de confianza 90%
        fig.add_trace(go.Scatter(
            x=forecast_df['Fecha'],
            y=forecast_df['ESM-hi-90'],
            mode='lines',
            name='IC 90% Superior',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['Fecha'],
            y=forecast_df['ESM-lo-90'],
            mode='lines',
            name='Intervalo de Confianza 90%',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(241, 143, 1, 0.4)',
            hovertemplate='<b>Fecha:</b> %{x}<br><b>IC 90%:</b> $%{y:,.0f}M COP<extra></extra>'
        ))
    
    # Personalizar el gráfico
    fig.update_layout(
        title={
            'text': 'Pronóstico de Ingresos por Renovaciones - Modelo Híbrido ETS-SARIMAX',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Arial Black'}
        },
        xaxis_title='Fecha',
        yaxis_title='Ingresos (Millones COP)',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1,
            tickformat='$,.0f'
        ),
        height=600
    )
    
    return fig


def to_excel(df):
    """Convierte DataFrame a Excel en memoria"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Proyección Ingresos', index=False)
        
        # Obtener el workbook y worksheet para formatear
        workbook = writer.book
        worksheet = writer.sheets['Proyección Ingresos']
        
        # Ajustar ancho de columnas
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    return output.getvalue()

