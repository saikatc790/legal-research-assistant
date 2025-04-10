import chromadb

chroma_client = chromadb.PersistentClient(path="./backend/src/db/chroma")
collection = chroma_client.get_or_create_collection("legal_docs")

print("Total Documents in ChromaDB:", len(collection.get()["documents"]))