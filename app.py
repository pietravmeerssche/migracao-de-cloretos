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
