import chromadb
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from reranker import rerank_results
from query_expansion import expand_query

# Load embedding model
embedding_model = SentenceTransformer("BAAI/bge-base-en")

#Setup Chroma DB
chroma_client = chromadb.PersistentClient(path="./backend/src/db/chroma")
collection = chroma_client.get_or_create_collection("legal_docs")

#load faiss index
faiss_index = faiss.read_index("D:/Legal Research Assistant/backend/src/db/faiss_index/legal.index")

# Fetch all stored documents from ChromaDB
stored_data = collection.get()
docs = stored_data["documents"]
metadata = stored_data["metadatas"]

# Setup BM25 for lexical search
bm25 = BM25Okapi([doc.split() for doc in docs])

def retrieve_relevant_chunks(query, top_k=5):
    """Retrieve relevant chunks using BM25, FAISS, and ChromaDB, then rerank."""

    #Query Expansion
    expanded_query = expand_query(query)

    #BM25 Lexical Search
    bm25_results = bm25.get_top_n(expanded_query.split(), docs, n=top_k)

    #ChromaDB Semantic Search
    chroma_results = collection.query(
        query_embeddings=[embedding_model.encode(expanded_query)],
        n_results=top_k
    )
    chroma_results = chroma_results["documents"][0]  

    #FAISS Semantic Search
    query_embedding = embedding_model.encode([expanded_query])[0]
    _, indices = faiss_index.search(np.array([query_embedding]), k=top_k)
    faiss_results = [docs[i] for i in indices[0]]  

    combined_results = list(set(bm25_results + chroma_results + faiss_results))
    reranked_chunks = rerank_results(expanded_query, combined_results)
    citations = []

    return reranked_chunks, citations
