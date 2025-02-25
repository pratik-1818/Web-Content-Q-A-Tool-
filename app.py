import requests
from bs4 import BeautifulSoup
import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load free Hugging Face embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_url(url, max_chars=5000):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return text[:max_chars]  # Limit content length
    return None

def query_vector_db(question, text):
    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    
    # Generate embeddings and store in FAISS
    vector_db = FAISS.from_texts(chunks, embedding_model)
    
    # Retrieve relevant context
    docs = vector_db.similarity_search(question, k=2)
    context = " ".join([doc.page_content for doc in docs])
    
    return context

def ask_groq(question, context):
    chat = ChatGroq(groq_api_key="api key")  # Replace with actual key
    response = chat.invoke([HumanMessage(content=f"Context: {context}\nQuestion: {question}")])
    return response.content

st.title("Web Content Q&A Tool")
url = st.text_input("Enter the URL:")
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if url and question:
        text = extract_text_from_url(url)
        if text:
            filtered_context = query_vector_db(question, text)  # Retrieve only relevant chunks
            answer = ask_groq(question, filtered_context)
            st.write("### Answer:")
            st.write(answer)
        else:
            st.error("Failed to extract content from the URL.")
    else:
        st.warning("Please enter both a URL and a question.")
