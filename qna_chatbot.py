import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

def extract_text_from_url(url, max_chars=4000):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return text[:max_chars]  # Limit content length to avoid exceeding token limits
    else:
        return None

def ask_groq(question, context):
    groq_api_key = os.getenv("GROQ_API_KEY")  # Get API key from environment variable
    if not groq_api_key:
        return "Error: GROQ_API_KEY is not set in the environment variables."
    chat = ChatGroq(groq_api_key=groq_api_key)
    response = chat.invoke([HumanMessage(content=f"Context: {context}\nQuestion: {question}")])
    return response.content

st.title("Web Content Q&A Tool")
url = st.text_input("Enter the URL:")
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if url and question:
        context = extract_text_from_url(url)
        if context:
            answer = ask_groq(question, context)
            st.write("### Answer:")
            st.write(answer)
        else:
            st.error("Failed to extract content from the URL.")
    else:
        st.warning("Please enter both a URL and a question.")
