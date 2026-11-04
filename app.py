from pathlib import Path
import traceback

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

st.set_page_config(
    page_title="Previsão de migração de cloretos",
    page_icon="🧪",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_DADOS = BASE_DIR / "databaseteste1.xlsx"

# A ordem abaixo reproduz exatamente as 14 colunas usadas no notebook
# (df.iloc[:, 0:14]); a coluna de resposta é a 15ª (dnssm).
FEATURES = [
    "wb", "idade", "cimento", "silica", "cinza", "escoria",
    "calcario", "metacaulim", "agregado", "sio2", "al2O3",
    "fe2O3", "cao", "mgo",
]
TARGET = "dnssm"

NOMES = {
    "wb": "Relação água/aglomerante",
    "idade": "Idade do ensaio em dias",
    "cimento": "Consumo de cimento em kg/m³",
    "silica": "Consumo de sílica em kg/m³",
    "cinza": "Consumo de cinza volante em kg/m³",
    "escoria": "Consumo de escória em kg/m³",
    "calcario": "Consumo de calcário em kg/m³",
    "metacaulim": "Consumo de metacaulim em kg/m³",
    "agregado": "Consumo total de agregados em kg/m³",
    "sio2": "Óxido de silício em %",
    "al2O3": "Óxido de alumínio em %",
    "fe2O3": "Óxido férrico em %",
    "cao": "Óxido de cálcio em %",
    "mgo": "Óxido de magnésio em %",
}

# Ordem de apresentação solicitada pelo usuário. A ordem interna FEATURES
# permanece inalterada para preservar a correspondência com o treinamento.
ORDEM_INTERFACE = [
    "wb", "cimento", "cinza", "calcario", "agregado", "idade",
    "silica", "escoria", "metacaulim", "sio2", "fe2O3", "mgo",
    "al2O3", "cao",
]

st.title("🧪 Previsão da migração de cloretos")
st.caption("Modelo de regressão XGBoost baseado na planilha e no notebook deste projeto.")
st.write(
    "Informe as características da mistura e do ensaio para estimar o coeficiente "
    "de migração de cloretos no estado não estacionário. "
    "A unidade do resultado é apresentada em × 10⁻¹² m²/s."
)


@st.cache_resource(show_spinner="Lendo a planilha e treinando o modelo XGBoost...")
def carregar_e_treinar(caminho: str, mtime_ns: int):
    """Lê a base e treina o modelo seguindo a divisão e os parâmetros do notebook."""
    del mtime_ns  # usado como parte da chave de cache para invalidar após trocar a planilha

    df = pd.read_excel(caminho, engine="openpyxl")
    if df.shape[1] < 15:
        raise ValueError(
            f"A planilha precisa ter pelo menos 15 colunas (14 entradas + DNSSM). "
            f"Foram encontradas {df.shape[1]} colunas."
        )

    # O notebook seleciona as colunas pela posição, não pelo nome.
    X = df.iloc[:, 0:14].copy()
    y = df.iloc[:, 14].copy()
    X.columns = FEATURES

    # Converte entradas e alvo para números; linhas incompletas/inválidas são excluídas.
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    validas = X.notna().all(axis=1) & y.notna()
    removidas = int((~validas).sum())
    X = X.loc[validas].astype(float)
    y = y.loc[validas].astype(float)

    if len(X) < 5:
        raise ValueError(
            f"Depois de remover linhas vazias ou não numéricas, restaram apenas {len(X)} registros."
        )

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    # Mesmos hiperparâmetros usados no notebook do repositório.
    modelo = XGBRegressor(
