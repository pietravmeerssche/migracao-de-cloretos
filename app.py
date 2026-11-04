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
        n_estimators=200,
        max_depth=2,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=0,
        n_jobs=1,
    )
    modelo.fit(X_treino, y_treino)

    r2_teste = float(modelo.score(X_teste, y_teste)) if len(X_teste) >= 2 else None
    estatisticas = {
        feature: {
            "min": float(X_treino[feature].min()),
            "max": float(X_treino[feature].max()),
            "mediana": float(X_treino[feature].median()),
        }
        for feature in FEATURES
    }

    return modelo, estatisticas, len(X), removidas, r2_teste, len(X_teste)


if not ARQUIVO_DADOS.is_file():
    st.error(
        "Não encontrei `databaseteste1.xlsx`. Coloque esse arquivo na mesma pasta "
        "de `app.py` (raiz do repositório do GitHub)."
    )
    st.stop()

try:
    modelo, estatisticas, n_registros, n_removidos, r2_teste, n_teste = carregar_e_treinar(
        str(ARQUIVO_DADOS), ARQUIVO_DADOS.stat().st_mtime_ns
    )
except Exception as erro:
    st.error("Não foi possível preparar a base ou treinar o XGBoost.")
    st.exception(erro)
    st.stop()

st.success("Modelo XGBoost treinado e pronto para realizar previsões.")

with st.expander("Informações do treinamento", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros válidos", n_registros)
    c2.metric("Registros no teste", n_teste)
    c3.metric("Linhas ignoradas", n_removidos)
    if r2_teste is not None:
        st.metric("R² no conjunto de teste", f"{r2_teste:.4f}")
    st.caption(
        "Divisão treino/teste: 70%/30%, random_state=0. "
        "Hiperparâmetros: 200 árvores, profundidade 2 e learning_rate 0,05."
    )

st.subheader("Dados de entrada")
st.caption("Os valores iniciais são as medianas do conjunto de treinamento. "
           "Os limites exibidos correspondem ao intervalo observado na base de treino.")

with st.form("formulario_dnssm"):
    valores = {}
    colunas = st.columns(2)

    for i, feature in enumerate(ORDEM_INTERFACE):
        stats = estatisticas[feature]
        minimo = stats["min"]
        maximo = stats["max"]
        padrao = stats["mediana"]

        # Evita intervalo degenerado quando uma variável tem valor constante.
        if minimo == maximo:
            margem = max(abs(minimo) * 0.01, 0.01)
            minimo_campo = minimo - margem
            maximo_campo = maximo + margem
        else:
            minimo_campo = minimo
            maximo_campo = maximo

        # Usamos text_input em vez de number_input para contornar a falha
        # de carregamento do módulo JavaScript NumberInput no navegador.
        valores[feature] = colunas[i % 2].text_input(
            NOMES.get(feature, feature),
            value=f"{padrao:.4f}",
            key=f"entrada_{feature}",
            help=(
                f"Digite um número. Intervalo observado no treino: "
                f"{stats['min']:.4g} a {stats['max']:.4g}. "
                "Você pode usar ponto ou vírgula decimal."
            ),
        )

    enviar = st.form_submit_button(
        "Estimar o coeficiente de migração de cloretos no estado não estacionário (× 10⁻¹² m²/s)",
        type="primary",
        use_container_width=True,
    )

if enviar:
    try:
        # Converte os campos de texto para números; aceita vírgula decimal.
        valores_numericos = {}
        for feature in FEATURES:
            texto = str(valores[feature]).strip().replace(",", ".")
            if not texto:
                raise ValueError(f"Preencha o campo: {NOMES.get(feature, feature)}")
            numero = float(texto)
            if not pd.notna(numero) or numero in (float("inf"), float("-inf")):
                raise ValueError(f"Informe um número finito em: {NOMES.get(feature, feature)}")
            valores_numericos[feature] = numero

        entrada = pd.DataFrame(
            [[valores_numericos[f] for f in FEATURES]], columns=FEATURES
        )
        previsao = float(modelo.predict(entrada)[0])

        st.success("Previsão calculada com sucesso.")
        st.metric(
            "Coeficiente de migração de cloretos no estado não estacionário",
            f"{previsao:.6g} × 10⁻¹² m²/s",
        )
        st.caption("Unidade apresentada conforme especificação da interface: × 10⁻¹² m²/s.")

        st.write("**Valores enviados ao modelo**")
        for feature in ORDEM_INTERFACE:
            st.write(f"- **{NOMES.get(feature, feature)}:** {valores_numericos[feature]:.4f}")

        fora_faixa = [
            feature for feature in FEATURES
            if valores_numericos[feature] < estatisticas[feature]["min"]
            or valores_numericos[feature] > estatisticas[feature]["max"]
        ]
        if fora_faixa:
            st.warning(
                "Uma ou mais entradas estão fora do intervalo observado no treinamento. "
                "A previsão pode envolver extrapolação."
            )

    except Exception as erro:
        st.error("Ocorreu um erro durante a previsão.")
        st.exception(erro)

st.divider()
st.caption(
    "Aplicação educacional baseada na base databaseteste1.xlsx e no script teste1.py. "
    "A interpretação física, as unidades e a aplicabilidade experimental do DNSSM "
    "devem ser confirmadas na documentação do conjunto de dados e do ensaio."
)
