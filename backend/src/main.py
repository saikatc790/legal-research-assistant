from fastapi import FastAPI, HTTPException
from retrieval import retrieve_relevant_chunks
from model import generate_answer
from query_expansion import expand_query

app = FastAPI()

@app.post("/query/")
async def query_legal_docs(query: str):
    try:
        expanded_query = expand_query(query)
        print(f"🔍 Expanded Query: {expanded_query}")

        relevant_chunks, citations = retrieve_relevant_chunks(expanded_query)

        if not relevant_chunks:
            raise HTTPException(status_code=404, detail="No relevant legal documents found.")

        response = generate_answer(query, relevant_chunks)

        return {
            # "query": query,
            "expanded_query": expanded_query,
            "response": response,
            "citations": citations
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
