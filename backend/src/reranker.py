from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-2-v2")

def rerank_results(query, retrieved_chunks):
    """Re-rank retrieved results based on query relevance."""

    pairs = [(query, chunk) for chunk in retrieved_chunks]

    scores = reranker.predict(pairs)

    ranked_results = sorted(zip(retrieved_chunks, scores), key=lambda x: x[1], reverse=True)

    reranked_chunks = [res[0] for res in ranked_results[:5]]

    return reranked_chunks

    