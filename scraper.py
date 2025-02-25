import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Set headers to bypass anti-bot mechanisms
HEADERS = {"User-Agent": "Mozilla/5.0"}

def scrape_text_from_url(url):
    """Fetches and extracts text from a given URL."""
    print(f"Fetching URL: {url}")
    response = requests.get(url, headers=HEADERS, timeout=10)  # Added timeout
    if response.status_code != 200:
        raise Exception(f"Failed to fetch page: {response.status_code}")

    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    
    # Extract first 5 paragraphs to avoid huge processing time
    text = '\n'.join([p.get_text() for p in paragraphs[:5]])  
    
    if not text.strip():
        raise Exception("No text extracted! Page might be blocking scraping.")
    
    return text

def generate_embedding(text):
    """Generates embeddings using Hugging Face's Sentence Transformer."""
    print("Generating embedding...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedding = model.encode([text])  # Convert text into embeddings
    return embedding

def store_in_faiss(embedding):
    """Stores embedding in a FAISS index."""
    print("Storing in FAISS...")
    dimension = embedding.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding)
    faiss.write_index(index, "vector_store.index")
    print("Embedding stored in FAISS index.")

def main():
    url = input("Enter URL to scrape: ")
    try:
        text = scrape_text_from_url(url)
        print(f"Text extracted ({len(text)} chars).")
        
        embedding = generate_embedding(text)
        print("Embedding generated successfully.")
        
        store_in_faiss(embedding)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
