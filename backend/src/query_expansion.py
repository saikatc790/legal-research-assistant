import requests
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions" 

def expand_query(query):
    """
    Expands the given legal query using LLaMA (Groq API) to improve retrieval.
    """
    prompt = f"""
    You are an AI assistant specializing in legal research.
    Expand the following legal query to improve search accuracy:

    Query: "{query}"

    Expanded Query:
    """

    response = requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
        },
    )

    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        print("Invalid JSON response:", response.text)
        return ""

    expanded_query = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    return expanded_query
