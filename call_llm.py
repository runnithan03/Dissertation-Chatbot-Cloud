import os
import requests

def llm_pipeline(prompt, max_new_tokens=150):
    headers = {
        "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-7b-instruct",
        "messages": [
            {"role": "system", "content": "You are an expert dissertation assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": max_new_tokens
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload
    )

    response.raise_for_status()  # Catch API errors
    return [{"generated_text": response.json()["choices"][0]["message"]["content"]}]
