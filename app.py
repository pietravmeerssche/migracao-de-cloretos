
import streamlit as st

st.set_page_config(
    page_title="Teste de entrada",
    page_icon="🧪"
)

st.title("Teste dos campos de entrada")

st.success("O Streamlit está funcionando!")

nome = st.text_input("Digite um valor de teste:")

if st.button("Enviar"):
    st.write(f"Valor recebido: {nome}")
    st.success("Campo de entrada funcionando!")
