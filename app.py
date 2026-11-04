from pathlib import Path
import math
import traceback

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

st.set_page_config(
    page_title="Previsão da migração de cloretos",
    page_icon="🧪",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_DADOS = BASE_DIR / "databaseteste1.xlsx"

# Ordem original utilizada pelo notebook: df.iloc[:, 0:14].
FEATURES = [
    "wb",
    "idade",
    "cimento",
    "silica",
    "cinza",
    "escoria",
    "calcario",
    "metacaulim",
    "agregado",
    "sio2",
    "al2O3",
    "fe2O3",
    "cao",
    "mgo",
]
TARGET_INDEX = 14

# Ordem solicitada para apresentação na interface. A ordem interna do
# modelo permanece FEATURES, exatamente como no treinamento do notebook.
ORDEM_INTERFACE = [
    "wb",
    "cimento",
    "cinza",
    "calcario",
    "agregado",
    "idade",
    "silica",
    "escoria",
    "metacaulim",
    "sio2",
    "fe2O3",
    "mgo",
    "al2O3",
    "cao",
]

NOMES = {
    "wb": "Relação água/aglomerante",
    "cimento": "Consumo de cimento em kg/m³",
    "cinza": "Consumo de cinza volante em kg/m³",
    "calcario": "Consumo de calcário em kg/m³",
    "agregado": "Consumo total de agregados em kg/m³",
    "idade": "Idade do ensaio em dias",
    "silica": "Consumo de sílica em kg/m³",
    "escoria": "Consumo de escória em kg/m³",
    "metacaulim": "Consumo de metacaulim em kg/m³",
    "sio2": "Óxido de silício em %",
    "fe2O3": "Óxido férrico em %",
    "mgo": "Óxido de magnésio em %",
    "al2O3": "Óxido de alumínio em %",
    "cao": "Óxido de cálcio em %",
}

st.title("🧪 Previsão da migração de cloretos")
st.caption("Modelo de regressão XGBoost")
st.write(
    "Informe a composição da mistura e as características do ensaio "
    "para estimar o coeficiente de migração de cloretos no estado "
    "não estacionário."
)
st.write("**Unidade do resultado:** × 10⁻¹² m²/s")


@st.cache_resource(show_spinner=False)
def carregar_e_treinar(caminho_arquivo: str, mtime_ns: int):
    """Lê a planilha e treina o XGBoost conforme o notebook original."""
    del mtime_ns  # Faz parte da chave do cache para detectar troca da planilha.

    print("[APP] Lendo databaseteste1.xlsx...", flush=True)
    dados = pd.read_excel(caminho_arquivo, engine="openpyxl")

    if dados.shape[1] < 15:
        raise ValueError(
            "A planilha precisa conter pelo menos 15 colunas: "
            "14 variáveis de entrada e a variável-alvo dnssm. "
            f"Foram encontradas {dados.shape[1]} colunas."
        )

    # O notebook seleciona as variáveis pela posição das colunas.
    X_df = dados.iloc[:, 0:14].copy()
    y_series = dados.iloc[:, TARGET_INDEX].copy()
    X_df.columns = FEATURES

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    y_series = pd.to_numeric(y_series, errors="coerce")

    # Registros sem alvo numérico não podem participar do treinamento.
    linhas_com_alvo = y_series.notna() & np.isfinite(y_series.astype(float))
    removidas = int((~linhas_com_alvo).sum())
    X_df = X_df.loc[linhas_com_alvo]
    y_series = y_series.loc[linhas_com_alvo]

    if len(X_df) < 5:
        raise ValueError(
            "Há menos de cinco registros com valores válidos na variável-alvo. "
            "Não é possível realizar a divisão treino/teste com segurança."
        )

    sem_dados = [feature for feature in FEATURES if X_df[feature].notna().sum() == 0]
    if sem_dados:
        raise ValueError(
            "Estas variáveis não possuem nenhum valor numérico na planilha: "
            + ", ".join(sem_dados)
        )

    X = X_df.to_numpy(dtype=float)
    y = y_series.to_numpy(dtype=float)

    print(f"[APP] Registros utilizáveis: {len(X)}. Iniciando divisão treino/teste...", flush=True)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    print("[APP] Treinando XGBoost...", flush=True)
    modelo = XGBRegressor(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=0,
        n_jobs=1,
    )
    modelo.fit(X_treino, y_treino)
    print("[APP] Treinamento concluído.", flush=True)

    r2_teste = float(modelo.score(X_teste, y_teste)) if len(X_teste) >= 2 else None

    estatisticas = {}
    for indice, feature in enumerate(FEATURES):
        serie = X_df[feature].dropna()
        estatisticas[feature] = {
            "min": float(serie.min()),
            "max": float(serie.max()),
            "mediana": float(serie.median()),
        }

    return modelo, estatisticas, len(X), removidas, r2_teste


if not ARQUIVO_DADOS.is_file():
    st.error(
        "Não encontrei `databaseteste1.xlsx`. O arquivo precisa estar na "
        "mesma pasta de `app.py`, na raiz do repositório GitHub."
    )
    st.stop()

st.info("Preparando a base de dados e treinando o modelo XGBoost...")

try:
    modelo, estatisticas, n_registros, n_removidos, r2_teste = carregar_e_treinar(
        str(ARQUIVO_DADOS), ARQUIVO_DADOS.stat().st_mtime_ns
    )
except Exception as erro:
    print("[APP] Falha ao carregar a base ou treinar o modelo:", flush=True)
    print(traceback.format_exc(), flush=True)
    st.error("Não foi possível preparar a base ou treinar o XGBoost.")
    st.exception(erro)
    st.stop()

st.success("Modelo XGBoost treinado e pronto para realizar previsões!")
st.write(f"Registros utilizados no treinamento: **{n_registros}**")
if n_removidos:
    st.warning(
        f"{n_removidos} linha(s) foram ignoradas porque a variável-alvo "
        "não continha um valor numérico."
    )
if r2_teste is not None:
    st.write(f"R² no conjunto de teste: **{r2_teste:.4f}**")
st.caption(
    "Configuração reproduzida do notebook: divisão treino/teste de 70%/30%, "
    "random_state=0, 200 árvores, profundidade 2 e learning_rate=0,05."
)

st.subheader("Dados de entrada")
st.caption(
    "Os campos começam preenchidos com a mediana observada na base. "
    "Você pode usar ponto ou vírgula como separador decimal."
)

valores_texto = {}
for feature in ORDEM_INTERFACE:
    mediana = estatisticas[feature]["mediana"]
    valores_texto[feature] = st.text_input(
        NOMES[feature],
        value=f"{mediana:.6g}",
        key=f"entrada_{feature}",
        help=(
            f"Intervalo observado na base: "
            f"{estatisticas[feature]['min']:.6g} a "
            f"{estatisticas[feature]['max']:.6g}."
        ),
    )

if st.button(
    "Estimar o coeficiente de migração de cloretos no estado não estacionário",
    type="primary",
):
    try:
        valores_numericos = {}
        for feature in FEATURES:
            texto = str(valores_texto[feature]).strip().replace(",", ".")
            if not texto:
                raise ValueError(f"Preencha o campo: {NOMES[feature]}.")
            try:
                numero = float(texto)
            except ValueError as exc:
                raise ValueError(
                    f"O valor informado em '{NOMES[feature]}' não é numérico."
                ) from exc
            if not math.isfinite(numero):
                raise ValueError(
                    f"Informe um número finito em '{NOMES[feature]}'."
                )
            valores_numericos[feature] = numero

        # A ordem é a mesma utilizada no treinamento original.
        entrada = np.array(
            [[valores_numericos[feature] for feature in FEATURES]],
            dtype=float,
        )
        previsao = float(modelo.predict(entrada)[0])

        st.success("Previsão calculada com sucesso!")
        st.write("### Coeficiente estimado")
        st.subheader(f"{previsao:.6g} × 10⁻¹² m²/s")
        st.write("**Valores informados:**")
        for feature in ORDEM_INTERFACE:
            st.write(f"- **{NOMES[feature]}:** {valores_numericos[feature]:.6g}")

        fora_faixa = [
            feature
            for feature in FEATURES
            if valores_numericos[feature] < estatisticas[feature]["min"]
            or valores_numericos[feature] > estatisticas[feature]["max"]
        ]
        if fora_faixa:
            st.warning(
                "Atenção: uma ou mais entradas estão fora do intervalo observado "
                "na base de treinamento. A previsão pode envolver extrapolação."
            )

    except Exception as erro:
        st.error("Não foi possível realizar a previsão.")
        st.exception(erro)

st.divider()
st.caption(
    "Aplicação educacional baseada na base databaseteste1.xlsx e no script teste1.py. "
    "Confirme a unidade e a interpretação físico-química do resultado na documentação "
    "do conjunto de dados e do ensaio."
)
