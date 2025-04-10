import requests
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def generate_answer(query, relevant_chunks):
    """
    Calls the Groq API to generate a legally accurate response.
    
    Args:
        query (str): User's legal question.
        relevant_chunks (list): List of retrieved relevant document passages.

    Returns:
        str: The generated legal response.
    """
    
    if not GROQ_API_KEY:
        raise ValueError("❌ Missing GROQ_API_KEY in .env file")

    # Format retrieved chunks properly
    formatted_chunks = "\n\n".join([f"- {chunk}" for chunk in relevant_chunks])

    # Construct API prompt
    messages = [
        {"role": "system", "content": "You are an expert legal assistant. Provide precise, well-structured, and fact-based legal answers."},
        {"role": "user", "content": f"Legal Question: {query}\n\nRelevant Information:\n{formatted_chunks}\n\nProvide a legally accurate summary with citations."}
    ]

    # API Request
    try:
        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama3-8b-8192", "messages": messages, "max_tokens": 512}
        )
        response.raise_for_status()  # Raises HTTPError if status_code is 4xx/5xx

        # Extract the response text
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    except requests.exceptions.RequestException as e:
        return f"❌ Error contacting Groq API: {str(e)}"
