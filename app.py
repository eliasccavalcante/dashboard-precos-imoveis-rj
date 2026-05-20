"""
Dashboard · Preços de Imóveis — Rio de Janeiro
Leia streamlit_data/ (gerado pelo notebook 01_simulacao_imoveis_rj.ipynb)
antes de rodar: streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Preços de Imóveis · Rio de Janeiro",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Paleta de cores ──────────────────────────────────────────────────────────
C_BLUE   = "#1B3A5C"
C_CORAL  = "#E8563A"
C_TEAL   = "#2A9D8F"
C_YELLOW = "#E9C46A"
C_LIGHT  = "#F0F4F8"
C_MID    = "#8DA9C4"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#2D3748"),
    margin=dict(l=10, r=10, t=40, b=10),
    title_font_size=15,
    title_font_color=C_BLUE,
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {C_LIGHT};
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background-color: white;
    padding: 6px 12px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}}

.stTabs [data-baseweb="tab"] {{
    height: 40px;
    padding: 0 18px;
    border-radius: 8px;
    font-weight: 500;
    font-size: 13px;
    color: {C_BLUE};
    background-color: transparent;
}}

.stTabs [aria-selected="true"] {{
    background-color: {C_BLUE} !important;
    color: white !important;
}}

div[data-testid="metric-container"] {{
    background-color: white;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    border-left: 4px solid {C_TEAL};
}}

div[data-testid="metric-container"] > label {{
    font-size: 12px !important;
    color: {C_MID} !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}

div[data-testid="metric-container"] > div {{
    font-size: 22px !important;
    font-weight: 700 !important;
    color: {C_BLUE} !important;
}}

.block-container {{
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}}

.section-header {{
    font-size: 13px;
    font-weight: 600;
    color: {C_MID};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 16px 0 8px 0;
}}
</style>
""", unsafe_allow_html=True)

# ── Cabeçalho ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,{C_BLUE},{C_TEAL});
            border-radius:14px;padding:28px 32px;margin-bottom:24px;">
  <h1 style="color:white;margin:0;font-size:26px;font-weight:700;">
    🏠 Preços de Imóveis — Rio de Janeiro
  </h1>
  <p style="color:rgba(255,255,255,0.80);margin:6px 0 0;font-size:14px;">
    Dataset simulado · 15.000 imóveis residenciais · LightGBM + Ridge · SHAP
  </p>
</div>
""", unsafe_allow_html=True)

# ── Funções de carregamento ──────────────────────────────────────────────────
DATA = Path("streamlit_data")


@st.cache_data
def load_json(path):
    with open(DATA / path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_parquet(path):
    return pd.read_parquet(DATA / path)


def fmt_brl(v):
    if v >= 1_000_000:
        return f"R$ {v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"R$ {v/1_000:.0f}K"
    return f"R$ {v:.0f}"


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗂️  Visão Geral",
    "📊  Análise Exploratória",
    "🔧  Feature Engineering",
    "🤖  Modelagem",
    "🔍  Interpretação",
])

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 · VISÃO GERAL DOS DADOS                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab1:
    try:
        dists = load_json("eda/driver_distributions.json")
        stats = load_parquet("eda/summary_stats.parquet")

        n_imoveis = int(stats.loc["count", "preco_venda"]) if "count" in stats.index else 15000
        preco_med = stats.loc["50%", "preco_venda"] if "50%" in stats.index else 0
        area_med  = stats.loc["50%", "area_util"]  if "50%" in stats.index else 1
        preco_m2  = preco_med / area_med if area_med > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Imóveis Simulados", f"{n_imoveis:,}")
        c2.metric("Preço Mediano",      fmt_brl(preco_med))
        c3.metric("Área Mediana",       f"{area_med:.0f} m²")
        c4.metric("Preço/m² Mediano",   fmt_brl(preco_m2))

        st.markdown("<div class='section-header'>Distribuições dos Atributos</div>",
                    unsafe_allow_html=True)

        driver_keys = [k for k in dists if k != "preco_venda"]
        for row_start in range(0, len(driver_keys), 3):
            cols = st.columns(3)
            for ci, key in enumerate(driver_keys[row_start:row_start + 3]):
                d = dists[key]
                color = C_TEAL if d["type"] == "histogram" else C_BLUE
                total = sum(d["y"]) or 1
                y_vals = [v / total * 100 for v in d["y"]]
                fig = go.Figure(go.Bar(
                    x=d["x"], y=y_vals,
                    marker_color=color, opacity=0.85,
                ))
                fig.update_layout(
                    **PLOTLY_LAYOUT, title=d["label"], height=220,
                    xaxis=dict(showgrid=False, tickfont_size=10),
                    yaxis=dict(showgrid=True, gridcolor="#EDF2F7",
                               tickfont_size=10, ticksuffix="%"),
                    bargap=0.12,
                )
                cols[ci].plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-header'>Estatísticas Descritivas</div>",
                    unsafe_allow_html=True)
        num_cols = [c for c in ["preco_venda", "area_util", "quartos", "suites",
                                "banheiros", "vagas", "andar", "idade_imovel"]
                    if c in stats.columns]
        st.dataframe(
            stats[num_cols].style.format("{:.1f}"),
            use_container_width=True, height=260,
        )

    except FileNotFoundError:
        st.warning("Execute `01_simulacao_imoveis_rj.ipynb` primeiro para gerar os artefatos.")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 · ANÁLISE EXPLORATÓRIA                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab2:
    try:
        price_dist  = load_json("eda/price_distribution.json")
        neigh_stats = load_parquet("eda/neighborhood_stats.parquet")
        tipo_stats  = load_parquet("eda/tipo_stats.parquet")

        st.markdown("<div class='section-header'>Distribuição de Preços de Venda</div>",
                    unsafe_allow_html=True)
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            pd_raw = price_dist["raw"]
            total_raw = sum(pd_raw["y"]) or 1
            y_raw_pct = [v / total_raw * 100 for v in pd_raw["y"]]
            fig = go.Figure(go.Bar(
                x=pd_raw["x"], y=y_raw_pct,
                marker_color=C_CORAL, opacity=0.80,
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT, title="Escala Original (R$)", height=280,
                xaxis=dict(tickformat=",.0f", showgrid=False, tickfont_size=10),
                yaxis=dict(showgrid=True, gridcolor="#EDF2F7",
                           tickfont_size=10, ticksuffix="%"),
            )
            col_d1.plotly_chart(fig, use_container_width=True)

        with col_d2:
            pd_log = price_dist["log"]
            total_log = sum(pd_log["y"]) or 1
            y_log_pct = [v / total_log * 100 for v in pd_log["y"]]
            fig = go.Figure(go.Bar(
                x=pd_log["x"], y=y_log_pct,
                marker_color=C_TEAL, opacity=0.80,
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT, title="Escala Log (log1p — target do modelo)", height=280,
                xaxis=dict(showgrid=False, tickfont_size=10),
                yaxis=dict(showgrid=True, gridcolor="#EDF2F7",
                           tickfont_size=10, ticksuffix="%"),
            )
            col_d2.plotly_chart(fig, use_container_width=True)

        # Ranking de bairros por preço/m²
        st.markdown("<div class='section-header'>Ranking de Bairros por Preço/m² Mediano</div>",
                    unsafe_allow_html=True)
        ns = neigh_stats.sort_values("preco_m2_mediano", ascending=True)
        fig_b = go.Figure(go.Bar(
            x=ns["preco_m2_mediano"], y=ns["bairro"],
            orientation="h",
            marker=dict(
                color=ns["preco_m2_mediano"],
                colorscale=[[0, C_MID], [0.5, C_TEAL], [1, C_CORAL]],
                showscale=True,
                colorbar=dict(title="R$/m²", thickness=12, len=0.6),
            ),
            text=[f"R$ {v:,.0f}" for v in ns["preco_m2_mediano"]],
            textposition="outside",
            textfont_size=10,
        ))
        fig_b.update_layout(
            **PLOTLY_LAYOUT, height=540,
            title="Preço Mediano por m² por Bairro",
            xaxis=dict(showgrid=True, gridcolor="#EDF2F7", tickformat=",.0f"),
            yaxis=dict(showgrid=False, tickfont_size=11),
        )
        st.plotly_chart(fig_b, use_container_width=True)

        col_t1, col_t2 = st.columns([1.3, 1])
        with col_t1:
            st.markdown("<div class='section-header'>Preço Mediano por Tipo de Imóvel</div>",
                        unsafe_allow_html=True)
            ts = tipo_stats.sort_values("median", ascending=False)
            fig_t = go.Figure(go.Bar(
                x=ts["tipo"], y=ts["median"],
                marker_color=[C_CORAL, C_TEAL, C_BLUE, C_YELLOW, C_MID][:len(ts)],
                text=[fmt_brl(v) for v in ts["median"]],
                textposition="outside",
            ))
            fig_t.update_layout(
                **PLOTLY_LAYOUT, height=300,
                title="Preço de Venda Mediano por Tipo",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#EDF2F7", tickformat=",.0f"),
            )
            col_t1.plotly_chart(fig_t, use_container_width=True)

        with col_t2:
            st.markdown("<div class='section-header'>Composição do Dataset</div>",
                        unsafe_allow_html=True)
            fig_pie = go.Figure(go.Pie(
                labels=ts["tipo"], values=ts["count"],
                hole=0.50,
                marker_colors=[C_BLUE, C_CORAL, C_TEAL, C_YELLOW, C_MID],
            ))
            fig_pie.update_layout(
                **PLOTLY_LAYOUT, height=300,
                title="Volume por Tipo",
                legend=dict(orientation="h", y=-0.15, font_size=11),
            )
            col_t2.plotly_chart(fig_pie, use_container_width=True)

    except FileNotFoundError:
        st.warning("Execute `01_simulacao_imoveis_rj.ipynb` primeiro.")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 · FEATURE ENGINEERING                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab3:
    try:
        import folium
        from streamlit_folium import st_folium

        geo_df  = load_parquet("features/geo_features.parquet")
        corr_df = load_parquet("features/correlation_matrix.parquet")
        centers = load_parquet("features/cluster_centers.parquet")

        # ── Mapa folium de clusters geográficos ──────────────────────────────
        st.markdown("<div class='section-header'>Clustering Geográfico · K-Means (k=10)</div>",
                    unsafe_allow_html=True)

        CLUSTER_HEX = [
            "#1B3A5C", "#E8563A", "#2A9D8F", "#E9C46A", "#8DA9C4",
            "#6C3BAA", "#2EC4B6", "#FF9F1C", "#E63946", "#1D3557",
        ]

        map_sample = geo_df.sample(min(2000, len(geo_df)), random_state=1)

        m = folium.Map(
            location=[-22.95, -43.28],
            zoom_start=11,
            tiles="CartoDB positron",
        )

        # Centros dos clusters (marcadores brancos maiores)
        for _, crow in centers.iterrows():
            folium.CircleMarker(
                location=[crow["lat"], crow["lon"]],
                radius=9,
                color="white",
                fill=True,
                fill_color="white",
                fill_opacity=1.0,
                weight=2,
            ).add_to(m)

        # Imóveis coloridos por cluster
        for _, row in map_sample.iterrows():
            color = CLUSTER_HEX[int(row["geo_cluster"]) % len(CLUSTER_HEX)]
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                weight=0,
                tooltip=(
                    f"<b>{row['bairro']}</b><br>"
                    f"Cluster: {int(row['geo_cluster'])}<br>"
                    f"Preço: R$ {row['preco_venda']:,.0f}<br>"
                    f"R$/m²: R$ {row['preco_m2']:,.0f}"
                ),
            ).add_to(m)

        st_folium(m, use_container_width=True, height=480, returned_objects=[])

        # ── Helper: linha de regressão linear ────────────────────────────────
        def regression_trace(x_vals, y_vals, color=C_CORAL, name="Regressão"):
            coeffs = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
            y_line = np.polyval(coeffs, x_line)
            r2 = 1 - np.sum((y_vals - np.polyval(coeffs, x_vals)) ** 2) / \
                     np.sum((y_vals - y_vals.mean()) ** 2)
            return go.Scatter(
                x=x_line, y=y_line,
                mode="lines",
                line=dict(color=color, width=2.5, dash="solid"),
                name=f"{name}  (R²={r2:.3f})",
            )

        samp = geo_df.sample(min(3000, len(geo_df)), random_state=2)

        # ── Distância à praia vs preço ───────────────────────────────────────
        st.markdown(
            "<div class='section-header'>Distância à Praia vs Preço (feature-chave para o Rio)</div>",
            unsafe_allow_html=True,
        )
        fig_praia = go.Figure()
        fig_praia.add_trace(go.Scatter(
            x=samp["distancia_praia_km"], y=samp["preco_venda"],
            mode="markers",
            marker=dict(
                color=samp["preco_m2"], colorscale="Viridis",
                size=5, opacity=0.55, showscale=True,
                colorbar=dict(title="R$/m²", thickness=12),
            ),
            name="Imóveis",
            text=samp["bairro"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Dist. praia: %{x:.2f} km<br>"
                "Preço: R$ %{y:,.0f}<extra></extra>"
            ),
        ))
        fig_praia.add_trace(regression_trace(
            samp["distancia_praia_km"].values, samp["preco_venda"].values,
            color=C_CORAL,
        ))
        fig_praia.update_layout(
            **PLOTLY_LAYOUT, height=380,
            title="Distância à Praia Mais Próxima vs Preço de Venda",
            xaxis=dict(title="Distância à praia (km)",
                       showgrid=True, gridcolor="#EDF2F7"),
            yaxis=dict(title="Preço de Venda (R$)",
                       showgrid=True, gridcolor="#EDF2F7", tickformat=",.0f"),
            legend=dict(orientation="h", y=1.08, font_size=11),
        )
        st.plotly_chart(fig_praia, use_container_width=True)

        # ── Pressão de vizinhança vs preço ───────────────────────────────────
        st.markdown(
            "<div class='section-header'>Pressão de Vizinhança vs Preço Real</div>",
            unsafe_allow_html=True,
        )
        fig_pv = go.Figure()
        fig_pv.add_trace(go.Scatter(
            x=samp["pressao_vizinhanca"], y=samp["preco_venda"],
            mode="markers",
            marker=dict(color=C_TEAL, size=5, opacity=0.45),
            name="Imóveis",
            text=samp["bairro"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Pressão viz.: R$ %{x:,.0f}<br>"
                "Preço real: R$ %{y:,.0f}<extra></extra>"
            ),
        ))
        fig_pv.add_trace(regression_trace(
            samp["pressao_vizinhanca"].values, samp["preco_venda"].values,
            color=C_CORAL,
        ))
        fig_pv.update_layout(
            **PLOTLY_LAYOUT, height=340,
            title="Mediana dos 50 Vizinhos Mais Próximos vs Preço Real",
            xaxis=dict(title="Pressão de vizinhança (R$)",
                       showgrid=True, gridcolor="#EDF2F7", tickformat=",.0f"),
            yaxis=dict(title="Preço de Venda (R$)",
                       showgrid=True, gridcolor="#EDF2F7", tickformat=",.0f"),
            legend=dict(orientation="h", y=1.08, font_size=11),
        )
        st.plotly_chart(fig_pv, use_container_width=True)

        # ── Matriz de correlação ─────────────────────────────────────────────
        st.markdown("<div class='section-header'>Matriz de Correlação das Features Numéricas</div>",
                    unsafe_allow_html=True)
        labels = corr_df.columns.tolist()
        z = corr_df.values
        fig_corr = go.Figure(go.Heatmap(
            z=z, x=labels, y=labels,
            colorscale=[[0, C_CORAL], [0.5, "white"], [1, C_TEAL]],
            zmid=0, zmin=-1, zmax=1,
            text=np.round(z, 2),
            texttemplate="%{text}",
            textfont_size=9,
            hovertemplate="%{x} × %{y}: %{z:.3f}<extra></extra>",
        ))
        fig_corr.update_layout(
            **PLOTLY_LAYOUT, height=480,
            title="Correlações entre variáveis numéricas",
            xaxis=dict(tickangle=-40, tickfont_size=10),
            yaxis=dict(tickfont_size=10, autorange="reversed"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    except FileNotFoundError:
        st.warning("Execute `01_simulacao_imoveis_rj.ipynb` primeiro.")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 4 · MODELAGEM                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab4:
    try:
        metrics_df = load_parquet("modeling/metrics.parquet")
        pred_df    = load_parquet("modeling/predictions.parquet")
        err_bin    = load_parquet("modeling/error_by_bin.parquet")
        err_bairro = load_parquet("modeling/error_by_bairro.parquet")
        cv_scores  = load_parquet("modeling/cv_scores.parquet")

        st.markdown("<div class='section-header'>Métricas Comparativas · Espaço Original (R$)</div>",
                    unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#EFF6FF;border-left:4px solid {C_TEAL};border-radius:8px;
                    padding:14px 18px;margin-bottom:16px;font-size:13px;line-height:1.7;
                    color:#2D3748;">
          <strong style="color:{C_BLUE};">Guia de métricas</strong> — todas avaliadas no espaço original de R$ (após reverter o log).<br>
          <strong>MAE</strong> (Mean Absolute Error) — erro absoluto médio: valor típico de desvio entre predito e real.<br>
          <strong>MdAE</strong> (Median Absolute Error) — versão mediana do MAE; menos sensível a imóveis atípicos caros.<br>
          <strong>RMSE</strong> (Root Mean Squared Error) — penaliza erros grandes mais do que o MAE; útil para detectar predições muito fora do real.<br>
          <strong>MAPE</strong> (Mean Absolute Percentage Error) — erro percentual médio; facilita comparação entre faixas de preço distintas.<br>
          <strong>R²</strong> — proporção da variância do preço explicada pelo modelo (1.0 = perfeito; 0 = equivale à média).
        </div>
        """, unsafe_allow_html=True)

        for _, row in metrics_df.iterrows():
            is_lgbm = "LightGBM" in str(row.get("modelo", ""))
            color   = C_TEAL if is_lgbm else C_MID
            st.markdown(
                f"<div style='background:white;border-radius:10px;padding:12px 20px;"
                f"border-left:4px solid {color};margin-bottom:8px;'>"
                f"<strong style='color:{C_BLUE};font-size:14px;'>{row.get('modelo','')}</strong>"
                f"</div>",
                unsafe_allow_html=True,
            )
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("MAE",     fmt_brl(row.get("MAE", 0)))
            c2.metric("MdAE",    fmt_brl(row.get("MdAE", 0)))
            c3.metric("RMSE",    fmt_brl(row.get("RMSE", 0)))
            c4.metric("MAPE",    f"{row.get('MAPE_%', 0):.1f}%")
            c5.metric("R²",      f"{row.get('R2', 0):.4f}")

        # Scatter predito vs real
        st.markdown("<div class='section-header'>Predito vs Real — test set</div>",
                    unsafe_allow_html=True)
        samp = pred_df.sample(min(2000, len(pred_df)), random_state=3)
        vmin, vmax = float(samp["y_true"].min()), float(samp["y_true"].max())

        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=samp["y_true"], y=samp["y_pred_lgbm"],
            mode="markers",
            marker=dict(color=C_TEAL, size=5, opacity=0.50),
            name="LightGBM",
            text=samp["bairro"],
            hovertemplate="<b>%{text}</b><br>Real: R$%{x:,.0f}<br>Pred: R$%{y:,.0f}<extra></extra>",
        ))
        fig_s.add_trace(go.Scatter(
            x=samp["y_true"], y=samp["y_pred_ridge"],
            mode="markers",
            marker=dict(color=C_MID, size=4, opacity=0.35),
            name="Ridge",
        ))
        fig_s.add_trace(go.Scatter(
            x=[vmin, vmax], y=[vmin, vmax],
            mode="lines",
            line=dict(color=C_CORAL, width=2, dash="dash"),
            name="Predição perfeita",
        ))
        fig_s.update_layout(
            **PLOTLY_LAYOUT, height=400,
            title="Preço Predito vs Preço Real",
            xaxis=dict(title="Real (R$)", tickformat=",.0f",
                       showgrid=True, gridcolor="#EDF2F7"),
            yaxis=dict(title="Predito (R$)", tickformat=",.0f",
                       showgrid=True, gridcolor="#EDF2F7"),
        )
        st.plotly_chart(fig_s, use_container_width=True)

        col_bin, col_bairro = st.columns(2)

        with col_bin:
            st.markdown("<div class='section-header'>MAE por Faixa de Preço (quintis)</div>",
                        unsafe_allow_html=True)
            fig_bin = go.Figure()
            fig_bin.add_trace(go.Bar(
                name="LightGBM",
                x=err_bin["faixa"].astype(str), y=err_bin["MAE_lgbm"],
                marker_color=C_TEAL, opacity=0.85,
            ))
            fig_bin.add_trace(go.Bar(
                name="Ridge",
                x=err_bin["faixa"].astype(str), y=err_bin["MAE_ridge"],
                marker_color=C_MID, opacity=0.85,
            ))
            fig_bin.update_layout(
                **PLOTLY_LAYOUT, height=300, barmode="group",
                title="Erro Absoluto Médio por Faixa de Preço",
                xaxis=dict(title="Quintil (Q1=mais barato)", showgrid=False),
                yaxis=dict(title="MAE (R$)", showgrid=True,
                           gridcolor="#EDF2F7", tickformat=",.0f"),
            )
            col_bin.plotly_chart(fig_bin, use_container_width=True)

        with col_bairro:
            st.markdown("<div class='section-header'>MAE LightGBM por Bairro</div>",
                        unsafe_allow_html=True)
            eb = err_bairro.sort_values("MAE_lgbm", ascending=True)
            fig_eb = go.Figure(go.Bar(
                x=eb["MAE_lgbm"], y=eb["bairro"],
                orientation="h",
                marker_color=C_CORAL, opacity=0.80,
                text=[fmt_brl(v) for v in eb["MAE_lgbm"]],
                textposition="outside",
                textfont_size=9,
            ))
            fig_eb.update_layout(
                **PLOTLY_LAYOUT, height=300,
                title="Erro por Bairro (test set)",
                xaxis=dict(showgrid=True, gridcolor="#EDF2F7", tickformat=",.0f"),
                yaxis=dict(showgrid=False, tickfont_size=10),
            )
            col_bairro.plotly_chart(fig_eb, use_container_width=True)

        # Validação cruzada
        st.markdown("<div class='section-header'>Validação Cruzada 5-Fold · LightGBM</div>",
                    unsafe_allow_html=True)
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Bar(
            name="MAE (R$)",
            x=cv_scores["fold"].astype(str), y=cv_scores["MAE"],
            marker_color=C_TEAL, opacity=0.85, yaxis="y",
        ))
        fig_cv.add_trace(go.Scatter(
            name="MAPE (%)",
            x=cv_scores["fold"].astype(str), y=cv_scores["MAPE_%"],
            mode="lines+markers",
            line=dict(color=C_CORAL, width=2),
            marker=dict(size=8),
            yaxis="y2",
        ))
        fig_cv.update_layout(
            **PLOTLY_LAYOUT, height=280,
            title="Estabilidade do Modelo por Fold de CV",
            xaxis=dict(title="Fold", showgrid=False),
            yaxis=dict(title="MAE (R$)", showgrid=True, gridcolor="#EDF2F7",
                       tickformat=",.0f", side="left"),
            yaxis2=dict(title="MAPE (%)", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_cv, use_container_width=True)

    except FileNotFoundError:
        st.warning("Execute `01_simulacao_imoveis_rj.ipynb` primeiro.")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 5 · INTERPRETAÇÃO SHAP                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab5:
    try:
        shap_imp = load_parquet("interpretation/shap_importance.parquet")
        shap_dep = load_json("interpretation/shap_dependence.json")
        impact   = load_json("interpretation/price_impact.json")

        # Importância global
        st.markdown(
            "<div class='section-header'>Importância Global das Features (Mean |SHAP value|)</div>",
            unsafe_allow_html=True,
        )
        top20 = shap_imp.head(20).sort_values("mean_abs_shap")
        fig_si = go.Figure(go.Bar(
            x=top20["mean_abs_shap"], y=top20["feature"],
            orientation="h",
            marker=dict(
                color=top20["mean_abs_shap"],
                colorscale=[[0, C_MID], [0.5, C_TEAL], [1, C_CORAL]],
                showscale=False,
            ),
            text=top20["mean_abs_shap"].round(4).astype(str),
            textposition="outside",
            textfont_size=10,
        ))
        fig_si.update_layout(
            **PLOTLY_LAYOUT, height=520,
            title="Top 20 Features por Importância SHAP — LightGBM",
            xaxis=dict(title="Mean |SHAP value|",
                       showgrid=True, gridcolor="#EDF2F7"),
            yaxis=dict(showgrid=False, tickfont_size=11),
        )
        st.plotly_chart(fig_si, use_container_width=True)

        st.markdown(f"""
        <div style="background:#EFF6FF;border-left:4px solid {C_TEAL};border-radius:8px;
                    padding:14px 18px;margin-bottom:8px;font-size:13px;line-height:1.7;
                    color:#2D3748;">
          <strong style="color:{C_BLUE};">Como interpretar</strong> — o gráfico mostra o <em>Mean |SHAP value|</em>
          de cada feature: a contribuição absoluta média que a variável gera na predição de log-preço ao longo de todos
          os imóveis do conjunto de teste. Quanto maior o valor, mais a feature desloca a predição do modelo em relação
          à média — para cima ou para baixo. Features no topo são as mais determinantes para o preço final.
        </div>
        """, unsafe_allow_html=True)

        # Dependence plots
        st.markdown("<div class='section-header'>Dependence Plots — Top Features</div>",
                    unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#EFF6FF;border-left:4px solid {C_TEAL};border-radius:8px;
                    padding:14px 18px;margin-bottom:8px;font-size:13px;line-height:1.7;
                    color:#2D3748;">
          <strong style="color:{C_BLUE};">Como interpretar</strong> — cada ponto é um imóvel do conjunto de teste.
          O eixo X é o valor real da feature; o eixo Y é o SHAP value daquela feature para aquele imóvel,
          ou seja, o quanto ela empurrou a predição para cima (positivo) ou para baixo (negativo) em relação
          à predição base. A cor também representa o SHAP value — tons quentes indicam maior impacto positivo
          no preço. Um padrão crescente da esquerda para a direita significa que valores maiores da feature
          elevam o preço predito.
        </div>
        """, unsafe_allow_html=True)

        dep_keys = list(shap_dep.keys())[:4]
        for row_start in range(0, len(dep_keys), 2):
            cols = st.columns(2)
            for ci, feat in enumerate(dep_keys[row_start:row_start + 2]):
                d = shap_dep[feat]
                fig_dep = go.Figure(go.Scatter(
                    x=d["x"], y=d["shap"],
                    mode="markers",
                    marker=dict(
                        color=d["shap"],
                        colorscale=[[0, C_BLUE], [0.5, C_TEAL], [1, C_CORAL]],
                        size=5, opacity=0.55, showscale=True,
                        colorbar=dict(title="SHAP", thickness=10, len=0.6),
                    ),
                    hovertemplate=f"{feat}: %{{x}}<br>SHAP: %{{y:.4f}}<extra></extra>",
                ))
                fig_dep.update_layout(
                    **PLOTLY_LAYOUT, height=280, title=f"SHAP Dependence — {feat}",
                    xaxis=dict(title=feat, showgrid=True,
                               gridcolor="#EDF2F7", tickfont_size=10),
                    yaxis=dict(title="SHAP value", showgrid=True,
                               gridcolor="#EDF2F7", tickfont_size=10),
                )
                cols[ci].plotly_chart(fig_dep, use_container_width=True)

        # Simulação de impacto no preço
        st.markdown(
            "<div class='section-header'>Impacto Marginal de Atributos no Preço de Venda</div>",
            unsafe_allow_html=True,
        )

        st.markdown(f"""
        <div style="background:#EFF6FF;border-left:4px solid {C_TEAL};border-radius:8px;
                    padding:14px 18px;margin-bottom:8px;font-size:13px;line-height:1.7;
                    color:#2D3748;">
          <strong style="color:{C_BLUE};">Como interpretar</strong> — o modelo é aplicado duas vezes sobre
          200 imóveis de treino sorteados, alterando apenas o atributo em análise e mantendo todo o restante fixo.
          A diferença média entre as predições estima o impacto isolado daquele atributo.<br>
          <strong>Atributos binários</strong> (Vista Mar, Piscina, etc.): compara predição com o atributo ativado (1) vs desativado (0).<br>
          <strong>Atributos de contagem</strong> (+1 Quarto, +1 Suíte, etc.): compara predição com valor original vs valor + 1 unidade.<br>
          As <strong>barras</strong> mostram a variação percentual média no preço (eixo esquerdo);
          os <strong>losangos</strong> mostram a variação em R$ absolutos (eixo direito).
          Barras em verde indicam impacto positivo; em vermelho, impacto negativo ou nulo.
        </div>
        """, unsafe_allow_html=True)
        sorted_impact = sorted(impact.items(), key=lambda x: x[1]["pct_delta"], reverse=True)
        labels_i = [k for k, _ in sorted_impact]
        pcts_i   = [v["pct_delta"]  for _, v in sorted_impact]
        deltas_i = [v["mean_delta"] for _, v in sorted_impact]

        # Padding de 40% acima/abaixo para acomodar rótulos externos
        pct_span  = max(abs(max(pcts_i, default=1)),  abs(min(pcts_i, default=-1))) or 1
        dlt_span  = max(abs(max(deltas_i, default=1)), abs(min(deltas_i, default=-1))) or 1
        pct_range  = [min(pcts_i)  - pct_span * 0.40, max(pcts_i)  + pct_span * 0.40]
        dlt_range  = [min(deltas_i) - dlt_span * 0.40, max(deltas_i) + dlt_span * 0.40]

        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            name="Δ% preço",
            x=labels_i, y=pcts_i,
            marker_color=[C_TEAL if p > 0 else C_CORAL for p in pcts_i],
            opacity=0.85, yaxis="y",
            text=[f"{p:+.1f}%" for p in pcts_i],
            textposition="outside", textfont_size=10,
        ))
        fig_imp.add_trace(go.Scatter(
            name="Δ R$ médio",
            x=labels_i, y=deltas_i,
            mode="markers",
            marker=dict(color=C_BLUE, size=10, symbol="diamond"),
            yaxis="y2",
        ))
        fig_imp.update_layout(
            **PLOTLY_LAYOUT, height=400,
            title="Impacto Marginal Médio de Cada Atributo Binário (simulação · 200 imóveis)",
            xaxis=dict(showgrid=False, tickangle=-20),
            yaxis=dict(title="Δ% no preço de venda", showgrid=True,
                       gridcolor="#EDF2F7", side="left", range=pct_range),
            yaxis2=dict(title="Δ R$ médio", overlaying="y", side="right",
                        showgrid=False, tickformat=",.0f", range=dlt_range),
            legend=dict(orientation="h", y=1.04),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        df_imp = pd.DataFrame([
            {"Atributo": k,
             "Δ R$ médio": fmt_brl(v["mean_delta"]),
             "Δ% preço": f"{v['pct_delta']:+.1f}%"}
            for k, v in sorted_impact
        ])
        st.dataframe(df_imp, use_container_width=True, hide_index=True)

    except FileNotFoundError:
        st.warning("Execute `01_simulacao_imoveis_rj.ipynb` primeiro.")
