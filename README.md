# Dashboard de Preços de Imóveis — Rio de Janeiro

Dashboard interativo para análise e modelagem de preços de venda de imóveis residenciais no Rio de Janeiro, construído com dados simulados e técnicas de machine learning interpretável.

---

## Visão geral

O projeto simula um dataset realista de **15.000 imóveis** distribuídos em 25 bairros do Rio de Janeiro, com preços por m² calibrados por região (de R$ 3.500/m² em Campo Grande a R$ 22.000/m² no Leblon). A partir desse dataset, é executado um pipeline completo de ciência de dados:

- Análise exploratória e distribuições
- Feature engineering com variáveis geoespaciais (distância à praia, clustering, pressão de vizinhança)
- Modelagem preditiva com **LightGBM** e **Ridge** como baseline
- Interpretabilidade com **SHAP**
- Visualização interativa via **Streamlit**

---

## Estrutura do projeto

```
├── 01_simulacao_imoveis_rj.ipynb   # Pipeline completo: dados → artefatos
├── app.py                          # Dashboard Streamlit
├── requirements.txt
└── streamlit_data/                 # Gerado pelo notebook (não versionado)
    ├── eda/
    ├── features/
    ├── modeling/
    ├── interpretation/
    └── models/
```

---

## Como executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Gerar os dados e artefatos

Execute todas as células do notebook **`01_simulacao_imoveis_rj.ipynb`** em ordem. Ele irá:

- Simular o dataset `imoveis_rj.csv`
- Gerar os artefatos de EDA, feature engineering, modelagem e SHAP em `streamlit_data/`
- Exportar o modelo treinado em `streamlit_data/models/lgbm_imoveis_rj.pkl`

### 3. Subir o dashboard

```bash
streamlit run app.py
```

---

## Abas do dashboard

| Aba | Conteúdo |
|---|---|
| 🗂️ Visão Geral | Distribuições de todos os atributos do dataset |
| 📊 Análise Exploratória | Preço por bairro, por tipo de imóvel, distribuição bruta e log |
| 🔧 Feature Engineering | Mapa de clusters geográficos, distância à praia vs preço, matriz de correlação |
| 🤖 Modelagem | Métricas LightGBM vs Ridge, predito × real, erro por faixa e por bairro, validação cruzada |
| 🔍 Interpretação | Importância SHAP, dependence plots, simulação de impacto marginal por atributo |

---

## Features de modelagem

| Feature | Descrição |
|---|---|
| `distancia_praia_km` | Distância à praia mais próxima (Copacabana, Ipanema, Leblon, Barra, Recreio) |
| `pressao_vizinhanca` | Mediana do preço dos 50 imóveis vizinhos mais próximos |
| `bairro_mediana` | Target encoding — mediana do preço de venda por bairro |
| `geo_cluster` | Cluster geográfico K-Means (k=10) sobre lat/lon |
| `log_area` | Log da área útil (normaliza a distribuição assimétrica) |
| `log_condominio` | Log do valor do condomínio mensal |
| `ratio_suites_quartos` | Proporção de suítes em relação ao total de quartos |
| `score_amenidades` | Soma de amenidades: varanda, piscina, academia, portaria 24h, vista mar |
| `novo_imovel` | Imóvel com até 3 anos de construção |
| `alto_andar` | Andar ≥ 10 |

---

## Stack

- **Dados & Modelagem**: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `shap`
- **Visualização**: `streamlit`, `plotly`, `folium`, `streamlit-folium`
- **Geoespacial**: `scikit-learn` (BallTree haversine, KMeans)

---

## Observação sobre os dados

O dataset é **inteiramente simulado** para fins didáticos. Os preços por m² e as distribuições de atributos foram calibrados para refletir o mercado imobiliário carioca, mas não representam transações reais.
