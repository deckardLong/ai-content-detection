import requests

API_BASE = 'http://localhost:8000'

def call_predict(text: str):
    resp = requests.post(f'{API_BASE}/predict', json={'text': text}, timeout=120)
    resp.raise_for_status()
    return resp.json()

def call_explain_llm(text: str):
    resp = requests.post(f'{API_BASE}/explain-llm', json={'text': text}, timeout=180)
    resp.raise_for_status()
    return resp.json()