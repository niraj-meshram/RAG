import streamlit as st
from main import vector_database, rag_chain

st.title("🧠🧠🧠Question and Answers🧠🧠🧠")
button = st.button("Create knowledgebase")
if button:
    pass

question = st.text_input("Question:")

if question:
    response = rag_chain.invoke(question)
    st.subheader("Answer:")
    st.write(response)