"""
FocusIA - Optimized Self-Evolving AI for Render Free Tier
Versione aggiornata con Supabase/PostgreSQL per memoria esterna
"""

import os
import threading
import requests
import numpy as np
from collections import deque
from flask import Flask, request, render_template_string, jsonify
from werkzeug.utils import secure_filename
import pypdf
import faiss
import psutil
import logging
import random
from psycopg2.extras import RealDictCursor
import psycopg2

# --- Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB max upload

context_memory = deque(maxlen=10)  # memoria leggera
context_lock = threading.Lock()
index = None
knowledge_chunks = []
embedding_cache = {}

logging.basicConfig(level=logging.INFO)

# --- Database helper ---
def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        cursor_factory=RealDictCursor
    )
    return conn

def init_database():
    conn = get_db_connection()
    c = conn.cursor()
    # knowledge table
    c.execute('''
        CREATE TABLE IF NOT EXISTS knowledge (
            id BIGSERIAL PRIMARY KEY,
            prompt TEXT,
            response TEXT,
            confidence REAL DEFAULT 1.0,
            usage_count INTEGER DEFAULT 0,
            category TEXT,
            embedding BYTEA,
            sentiment REAL,
            cluster INTEGER
        )
    ''')
    # brain_state table
    c.execute('''
        CREATE TABLE IF NOT EXISTS brain_state (
            key TEXT PRIMARY KEY,
            value REAL
        )
    ''')
    for key, val in [('age',0), ('curiosity',0.8), ('empathy',0.5), ('knowledge_clusters',3)]:
        c.execute('INSERT INTO brain_state (key, value) VALUES (%s,%s) ON CONFLICT (key) DO NOTHING', (key, val))
    # prompts table
    c.execute('''
        CREATE TABLE IF NOT EXISTS prompts (
            id BIGSERIAL PRIMARY KEY,
            system_prompt TEXT,
            score REAL DEFAULT 0
        )
    ''')
    base_prompt = "Sei FocusIA, un'IA evolutiva creata da Xhulio Guranjaku. Rispondi in modo professionale e utile."
    c.execute('INSERT INTO prompts (system_prompt, score) VALUES (%s,0) ON CONFLICT DO NOTHING', (base_prompt,))
    conn.commit()
    conn.close()

def save_interaction(prompt, response, confidence, category, sentiment, embedding):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO knowledge (prompt, response, confidence, category, embedding, sentiment)
        VALUES (%s, %s, %s, %s, %s, %s)
    ''', (prompt, response, confidence, category, embedding.tobytes(), sentiment))
    conn.commit()
    conn.close()

def get_brain_state():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT key, value FROM brain_state')
    state = {row['key']: row['value'] for row in c.fetchall()}
    conn.close()
    return state

def update_brain_state(age_inc=0.1, curiosity_adj=0, empathy_adj=0, clusters_adj=0):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE brain_state SET value = value + %s WHERE key='age'", (age_inc,))
    c.execute("UPDATE brain_state SET value = LEAST(GREATEST(value + %s,0),1) WHERE key='curiosity'", (curiosity_adj,))
    c.execute("UPDATE brain_state SET value = LEAST(GREATEST(value + %s,0),1) WHERE key='empathy'", (empathy_adj,))
    c.execute("UPDATE brain_state SET value = value + %s WHERE key='knowledge_clusters'", (clusters_adj,))
    conn.commit()
    conn.close()

# --- Embedding ---
def get_embedding(text):
    if text in embedding_cache:
        return embedding_cache[text]
    try:
        res = requests.post(
            "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
            headers={"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"},
            json={"inputs": text},
            timeout=5
        )
        emb = np.array(res.json()[0])
        embedding_cache[text] = emb
        return emb
    except:
        return np.zeros(384)

# --- RAG Index ---
def update_rag_index(new_chunks):
    global index, knowledge_chunks
    if index is None:
        index = faiss.IndexFlatL2(384)
    embeddings = np.array([get_embedding(c) for c in new_chunks])
    index.add(embeddings)
    knowledge_chunks.extend(new_chunks)

def retrieve_relevant_chunks(query, top_k=2):
    if index is None or index.ntotal == 0:
        return []
    query_emb = get_embedding(query)
    _, indices = index.search(np.array([query_emb]), top_k)
    return [knowledge_chunks[i] for i in indices[0] if i < len(knowledge_chunks)]

# --- Chatbot ---
def chatbot_response(prompt):
    global context_memory
    state = get_brain_state()
    context = "\n".join(context_memory)
    # risposta semplice
    response = "Sto imparando..."
    conf = random.uniform(0.7, 0.95)
    sent = 0.0
    emb = get_embedding(prompt)
    save_interaction(prompt, response, conf, "general", sent, emb)
    update_brain_state()
    context_memory.append(f"Utente: {prompt}\nFocusIA: {response}")
    return response

# --- PDF ---
def extract_text_from_pdf(filepath):
    text = ""
    with open(filepath, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages[:5]:
            text += page.extract_text() + "\n"
    chunks = [text[i:i+200] for i in range(0, len(text), 200)]
    return chunks

# --- HTML ---
HTML = """
<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<title>FocusIA Dashboard</title>
</head>
<body>
<h2>FocusIA Chat</h2>
<input type="text" id="user-input">
<button onclick="sendMessage()">Invia</button>
<div id="chat-box"></div>
<script>
async function sendMessage() {
    const input = document.getElementById('user-input');
    const res = await fetch('/chat', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({prompt: input.value})
    });
    const data = await res.json();
    document.getElementById('chat-box').innerHTML += "<div>"+data.response+"</div>";
    input.value='';
}
</script>
</body>
</html>
"""

# --- Routes ---
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data.get('prompt','')
    response = chatbot_response(prompt)
    return jsonify({"response": response})

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        chunks = extract_text_from_pdf(filepath)
        update_rag_index(chunks)
        return jsonify({"message": "PDF indicizzato"})
    return jsonify({"message": "Errore"}), 400

# --- Avvio ---
if __name__ == "__main__":
    init_database()
    app.run(host="0.0.0.0", port=5000)

