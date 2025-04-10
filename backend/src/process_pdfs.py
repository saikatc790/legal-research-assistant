import nltk
nltk.download('punkt_tab')
from embedding import create_and_store_embeddings, extract_and_chunk_pdfs

def process_pdfs():
    text_chunks, metadata = extract_and_chunk_pdfs()

    if text_chunks:
        print("Creating and storing embeddings...")
        create_and_store_embeddings(text_chunks, metadata)
        print("Embeddings created successfully!")
    else:
        print("No valid text extracted from PDFs!")

if __name__ == "__main__":
    process_pdfs()