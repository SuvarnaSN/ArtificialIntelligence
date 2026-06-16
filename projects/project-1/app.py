import streamlit as st
from chatbot import chatbot_response

st.title("Chatbot")

prompt = st.chat_input("Type a message")

if prompt:
    st.chat_message("user").write(prompt)

    response = chatbot_response(prompt)

    st.chat_message("assistant").write(response)
