const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function postJSON(path, body) {
    const res = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(body),
    });
    if (!res.ok) {
        throw new  Error(`${path} failed with status ${res.status}`);
    }
    return res.json();
}

export const predictText = (text) => postJSON('/predict', {text})
export const explainText = (text) => postJSON('/explain', {text})
export const explainLlmText = (text) => postJSON('/explain-llm', { text });