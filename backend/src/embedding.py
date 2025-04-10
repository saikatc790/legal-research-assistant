from sentence_transformers import SentenceTransformer
import chromadb
import faiss
import numpy as np
import fitz
import os
from nltk.tokenize import sent_tokenize

#load embedding model
embedding_model = SentenceTransformer("BAAI/bge-base-en")  

#Setup Chroma DB
chroma_client = chromadb.PersistentClient(path="./backend/src/db/chroma")
collection = chroma_client.get_or_create_collection("legal_docs")

#Setup FAISS index
dimension = embedding_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension)

# Folder path to PDFs
pdf_folder = "D:/Legal Research Assistant/backend/src/data/legal_docs/"
faiss_path = "D:/Legal Research Assistant/backend/src/db/faiss_index/legal.index"
os.makedirs(os.path.dirname(faiss_path), exist_ok=True)


# Function to split text into smaller chunks
def chunk_text(text, max_chunk_size=256, overlap=30):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], []
    current_length = 0

    for sentence in sentences:
        tokens = sentence.split()
        if current_length + len(tokens) > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:]  # Overlapping context
            current_length = sum(len(s.split()) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += len(tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Function to extract and chunk text from PDFs using PyMuPDF
def extract_and_chunk_pdfs():
    text_chunks = []
    metadata = []

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            text = ""

            try:
                doc = fitz.open(pdf_path)
                text = "\n".join([page.get_text("text") for page in doc])
            except Exception as e:
                print(f"Failed to extract text from {filename}: {e}")
                continue

            if text.strip():
                chunks = chunk_text(text)
                text_chunks.extend(chunks)
                metadata.extend([{"filename": filename}] * len(chunks))

    return text_chunks, metadata

def create_and_store_embeddings(text_chunks, metadata):
    print("🔹 Creating embeddings...")
    embeddings = embedding_model.encode(text_chunks, show_progress_bar=True)

    for idx, (chunk, meta) in enumerate(zip(text_chunks, metadata)):
        print(f"Storing chunk {idx} in ChromaDB & FAISS")  # Debugging
        collection.add(
            ids=[str(idx)],
            embeddings=[embeddings[idx]],
            documents=[chunk],
            metadatas=[meta]
        )
        index.add(np.array([embeddings[idx]]))

    print("Embeddings created and stored successfully!")

    # Save FAISS index after adding all embeddings
    faiss.write_index(index, "D:/Legal Research Assistant/backend/src/db/faiss_index/legal.index")
    print("FAISS index saved successfully.")


        
