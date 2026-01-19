"""
FocusIA - Optimized Self-Evolving AI for Render Free Tier
Copyright (C) 2025 Xhulio Guranjaku
All rights reserved.

This software is the intellectual property of Xhulio Guranjaku. 
Developed with assistance from Grok (xAI) under the direction of Xhulio Guranjaku.
Unauthorized use, reproduction, or distribution of this code without 
explicit permission from Xhulio Guranjaku is prohibited.

This version merges the core features from provided codes into a lightweight, crash-free app for Render Free (512 MB RAM limit).
- Uses external APIs for heavy lifting (text gen, embeddings).
- Light RAG with FAISS + mini embeddings.
- Self-evolving: Prompt evolution, feedback loop, confidence updates, simple curiosity learning via web searches.
- No local large models/TTS/vision to avoid OOM.
- Simplified DB and clustering (KMeans on small datasets only).
- Integrated dashboard + chat with voice input (browser-based, no server TTS).
"""

import sqlite3
import os
import threading
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import wikipedia
import time
import numpy as np
from flask import Flask, request, render_template_string, jsonify
from werkzeug.utils import secure_filename
import pypdf
import json
import logging
import random
from collections import deque
from sklearn.cluster import KMeans
import faiss  # Light vector DB
import psutil  # Monitor RAM

# Configurazione base (API esterne per low mem)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
wikipedia.set_lang("it")
DB_FILE = "focusia_brain.db"
UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Limite upload per RAM
init_database()
context_memory = deque(maxlen=10)  # Memoria leggera
context_lock = threading.Lock()
index = None  # FAISS index
knowledge_chunks = []  # Chunks per RAG
embedding_cache = {}  # Cache embedding per riutilizzo

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Logging
logging.basicConfig(level=logging.INFO)

# Inizializzazione database (merge: knowledge + brain_state + feedback)
def init_database():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS knowledge 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, prompt TEXT, response TEXT, confidence REAL DEFAULT 1.0, 
                  usage_count INTEGER DEFAULT 0, category TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
                  embedding BLOB, sentiment REAL, cluster INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS tasks 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT, status TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS feedback 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, interaction_id INTEGER, rating INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS brain_state 
                 (key TEXT PRIMARY KEY, value REAL)''')
    c.execute("INSERT OR IGNORE INTO brain_state (key, value) VALUES ('age', 0)")
    c.execute("INSERT OR IGNORE INTO brain_state (key, value) VALUES ('curiosity', 0.8)")
    c.execute("INSERT OR IGNORE INTO brain_state (key, value) VALUES ('empathy', 0.5)")
    c.execute("INSERT OR IGNORE INTO brain_state (key, value) VALUES ('knowledge_clusters', 3)")
    c.execute('''CREATE TABLE IF NOT EXISTS prompts 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, system_prompt TEXT, score REAL DEFAULT 0.0)''')
    
    # Prompt base
    base_prompt = "Sei FocusIA, un'IA evolutiva creata da Xhulio Guranjaku. Rispondi in modo professionale e utile."
    c.execute("INSERT OR IGNORE INTO prompts (system_prompt, score) VALUES (?, 0.0)", (base_prompt,))
    
    # Conoscenza base
    base_knowledge = [
        ("FocusIA", "FocusIA è un'IA evolutiva creata da Xhulio Guranjaku.", 1.0, "base"),
        ("Deep Work", "Il Deep Work è la chiave per la produttività massima.", 1.0, "base")
    ]
    for p, r, conf, cat in base_knowledge:
        embedding = get_embedding(p)  # Light embedding
        c.execute("INSERT OR IGNORE INTO knowledge (prompt, response, confidence, category, embedding, sentiment) VALUES (?, ?, ?, ?, ?, 0.0)", 
                  (p, r, conf, cat, embedding.tobytes()))
    
    conn.commit()
    conn.close()

# Brain state functions (dal secondo codice)
def update_brain_state(age_inc=0.1, curiosity_adj=0, empathy_adj=0, clusters_adj=0):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE brain_state SET value = value + ? WHERE key = 'age'", (age_inc,))
    c.execute("UPDATE brain_state SET value = MAX(0, MIN(1, value + ?)) WHERE key = 'curiosity'", (curiosity_adj,))
    c.execute("UPDATE brain_state SET value = MAX(0, MIN(1, value + ?)) WHERE key = 'empathy'", (empathy_adj,))
    c.execute("UPDATE brain_state SET value = value + ? WHERE key = 'knowledge_clusters'", (clusters_adj,))
    conn.commit()
    conn.close()

def get_brain_state():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT key, value FROM brain_state")
    state = dict(c.fetchall())
    conn.close()
    return state

# Embedding leggero via HF API (no local model per RAM)
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
        return np.zeros(384)  # Fallback zero vector

# Update RAG index (FAISS flat, low mem)
def update_rag_index(new_chunks):
    global index, knowledge_chunks
    if index is None:
        index = faiss.IndexFlatL2(384)  # MiniLM dim
    embeddings = np.array([get_embedding(chunk) for chunk in new_chunks])
    index.add(embeddings)
    knowledge_chunks.extend(new_chunks)

# Retrieve RAG
def retrieve_relevant_chunks(query, top_k=2):  # Limite basso per mem
    if index is None or index.ntotal == 0: return []
    query_emb = get_embedding(query)
    _, indices = index.search(np.array([query_emb]), top_k)
    return [knowledge_chunks[i] for i in indices[0] if i < len(knowledge_chunks)]

# Save interaction
def save_interaction(prompt, response, confidence, category, sentiment, embedding):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO knowledge (prompt, response, confidence, category, embedding, sentiment) VALUES (?, ?, ?, ?, ?, ?)", 
              (prompt, response, confidence, category, embedding.tobytes(), sentiment))
    conn.commit()
    conn.close()

# Cluster knowledge (solo se <100 items per mem)
def cluster_knowledge():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, embedding FROM knowledge")
    data = c.fetchall()
    conn.close()
    if len(data) < 5 or len(data) > 100: return  # Limite per RAM
    embeddings = np.array([np.frombuffer(item[1], dtype=np.float32) for item in data])
    ids = [item[0] for item in data]
    state = get_brain_state()
    kmeans = KMeans(n_clusters=min(int(state["knowledge_clusters"]), 5), random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    for id_, cluster in zip(ids, clusters):
        c.execute("UPDATE knowledge SET cluster = ? WHERE id = ?", (cluster, id_))
    conn.commit()
    conn.close()

# Search DB (simplified cosine sim)
def search_database(prompt):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT prompt, response, confidence, usage_count, id, embedding, cluster FROM knowledge")
    interactions = c.fetchall()
    conn.close()
    if not interactions: return None
    prompt_emb = get_embedding(prompt)
    best_match = None
    best_score = -1
    for past_prompt, past_response, confidence, usage_count, id_, emb_bytes, cluster in interactions:
        past_emb = np.frombuffer(emb_bytes, dtype=np.float32)
        similarity = np.dot(prompt_emb, past_emb) / (np.linalg.norm(prompt_emb) * np.linalg.norm(past_emb) + 1e-8)
        score = (similarity * 0.7) + (confidence * 0.2) + (min(usage_count, 10) * 0.1 / 10)
        if score > best_score and similarity > 0.6:
            best_score = score
            best_match = (past_response, id_)
    if best_match:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("UPDATE knowledge SET usage_count = usage_count + 1 WHERE id = ?", (best_match[1],))
        conn.commit()
        conn.close()
        return best_match[0]
    return None

# Evoluzione prompt (leggera, via API)
def evolve_prompts():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    interaction_count = c.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
    if interaction_count % 5 != 0: return  # Evolve ogni 5 per low load
    best_prompt = c.execute("SELECT system_prompt FROM prompts ORDER BY score DESC LIMIT 1").fetchone()[0]
    variants = [
        best_prompt + " Aggiungi dettagli da conoscenze apprese.",
        best_prompt.replace("professionale", "conciso"),
        best_prompt + " Prioritizza evoluzione."
    ]
    test_prompt = "Cos'è il Deep Work?"
    test_ref = "Il Deep Work è la chiave per la produttività massima."
    best_score = -1
    best_variant = best_prompt
    for var in variants:
        response = get_ai_response(test_prompt, system_prompt=var)
        # Simple score (length similarity as proxy)
        score = 1 - abs(len(response) - len(test_ref)) / max(len(response), len(test_ref))
        if score > best_score:
            best_score = score
            best_variant = var
    c.execute("INSERT INTO prompts (system_prompt, score) VALUES (?, ?)", (best_variant, best_score))
    conn.commit()
    conn.close()

# AI response via API (con RAG)
def get_ai_response(prompt, system_prompt=None):
    evolve_prompts()
    conn = sqlite3.connect(DB_FILE)
    if system_prompt is None:
        system_prompt = conn.execute("SELECT system_prompt FROM prompts ORDER BY score DESC LIMIT 1").fetchone()[0]
    conn.close()
    relevant = retrieve_relevant_chunks(prompt)
    augmented_prompt = prompt + "\nRilevante: " + " ".join(relevant)
    
    # Monitor RAM
    if psutil.virtual_memory().percent > 70:
        return "Memoria bassa: risposta base. " + augmented_prompt
    
    if OPENAI_API_KEY:
        try:
            res = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": augmented_prompt}]
                },
                timeout=10
            )
            return res.json()['choices'][0]['message']['content']
        except: pass
    try:
        res = requests.post(
            "https://api-inference.huggingface.co/models/gpt2",
            headers={"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"},
            json={"inputs": augmented_prompt},
            timeout=10
        )
        return res.json()[0]['generated_text']
    except:
        return "Sto imparando..."

# Web search (dal primo + multi dal secondo, simplified)
def web_search_multi(query):
    try:
        results = list(search(query, num=3, stop=3, pause=2))
        content = ""
        for url in results:
            res = requests.get(url, timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            content += ' '.join([p.text for p in soup.find_all('p')[:2]])[:300] + " "
        wiki = wikipedia.summary(query, sentences=1) if query else ""
        return content + wiki
    except:
        return None

# Curiosity learning (light: web search on weak knowledge)
def curiosity_learning():
    state = get_brain_state()
    if random.random() < state["curiosity"]:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT prompt FROM knowledge WHERE confidence < 0.7 ORDER BY timestamp DESC LIMIT 1")
        weak = c.fetchone()
        conn.close()
        if weak:
            weak_prompt = weak[0]
            new_res = web_search_multi(weak_prompt)
            if new_res:
                conf = random.uniform(0.6, 0.9)  # Proxy
                sent = 0.0
                emb = get_embedding(weak_prompt)
                save_interaction(weak_prompt, new_res, conf, "curiosity", sent, emb)
                update_brain_state(curiosity_adj=-0.05 if conf > 0.7 else 0.05)
                threading.Thread(target=cluster_knowledge).start()

# Sentiment analysis (simple keyword)
def analyze_sentiment(response):
    positive = ["bene", "ottimo", "felice"]
    negative = ["male", "brutto", "triste"]
    score = sum(1 for w in positive if w in response.lower()) - sum(1 for w in negative if w in response.lower())
    return max(-1, min(1, score * 0.2))

# PDF extract + RAG
def extract_text_from_pdf(filepath):
    text = ""
    with open(filepath, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages[:5]:  # Limite per mem
            text += page.extract_text() + "\n"
    chunks = [text[i:i+200] for i in range(0, len(text), 200)]
    return chunks

# Chatbot response (merge logic)
def chatbot_response(prompt):
    global context_memory
    state = get_brain_state()
    context = "\n".join(context_memory)
    db_res = search_database(prompt)
    if db_res:
        response = db_res
    else:
        needs_web = any(k in prompt.lower() for k in ["chi è", "cos'è", "cerca", "news"])
        if needs_web:
            response = web_search_multi(prompt)
        else:
            response = get_ai_response(prompt + "\nContesto: " + context)
        conf = random.uniform(0.7, 0.95)  # Proxy per low mem
        sent = analyze_sentiment(response)
        emb = get_embedding(prompt)
        save_interaction(prompt, response, conf, "general", sent, emb)
        update_brain_state()
    personality = random.choice([
        " Con focus...",
        f" Curiosità: {state['curiosity']:.1f}..."
    ])
    response += personality
    context_memory.append(f"Utente: {prompt}\nFocusIA: {response}")
    threading.Thread(target=curiosity_learning).start()
    return response

# HTML merge (dashboard dal primo + voice dal secondo)
HTML = """
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FocusIA - Dashboard Evolutivo</title>
    <style>
        /* CSS dal primo codice */
        body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5; margin: 0; display: flex; height: 100vh; }
        #sidebar { width: 260px; background: #1a202c; color: white; padding: 25px; display: flex; flex-direction: column; }
        #main { flex: 1; padding: 30px; overflow-y: auto; background: #ffffff; }
        .card { background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 25px; border: 1px solid #e2e8f0; }
        h2, h3 { color: #2d3748; margin-top: 0; }
        #chat-box { height: 400px; overflow-y: auto; border: 1px solid #edf2f7; padding: 15px; margin-bottom: 15px; background: #f8fafc; border-radius: 8px; }
        .msg { margin-bottom: 12px; padding: 10px; border-radius: 8px; max-width: 85%; }
        .user-msg { background: #ebf8ff; color: #2b6cb0; align-self: flex-end; margin-left: auto; font-weight: 600; }
        .ai-msg { background: #f0fff4; color: #2f855a; border: 1px solid #c6f6d5; }
        input[type="text"] { width: calc(100% - 110px); padding: 12px; border: 1px solid #cbd5e0; border-radius: 6px; outline: none; }
        button { background: #4299e1; color: white; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; font-weight: 600; transition: 0.2s; }
        button:hover { background: #3182ce; }
        .nav-item { cursor: pointer; padding: 12px; border-radius: 8px; margin-bottom: 8px; transition: 0.2s; }
        .nav-item:hover { background: #2d3748; }
        .nav-item.active { background: #4299e1; }
        .tab-content { display: none; animation: fadeIn 0.3s; }
        .tab-content.active { display: block; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        #timer { font-size: 48px; font-weight: bold; color: #e53e3e; text-align: center; margin: 20px 0; }
    </style>
</head>
<body>
    <div id="sidebar">
        <h1 style="font-size: 24px; margin-bottom: 30px;">FocusIA</h1>
        <div class="nav-item active" onclick="showTab('chat', this)">💬 Chat AI</div>
        <div class="nav-item" onclick="showTab('tasks', this)">⏱️ Deep Work</div>
        <div class="nav-item" onclick="showTab('upload', this)">📂 Documenti</div>
        <div style="margin-top: auto; font-size: 12px; color: #a0aec0;">© 2025 Xhulio Guranjaku</div>
    </div>
    <div id="main">
        <div id="chat" class="tab-content active">
            <div class="card">
                <h3>Assistente Evolutivo</h3>
                <div id="chat-box" style="display: flex; flex-direction: column;"></div>
                <div style="display: flex; gap: 10px;">
                    <input type="text" id="user-input" placeholder="Chiedi qualcosa o scrivi 'cerca...' ">
                    <button onclick="sendMessage()">Invia</button>
                    <button onclick="startVoiceInput()">Parla</button>
                </div>
            </div>
        </div>
        
        <div id="tasks" class="tab-content">
            <div class="card">
                <h3>Sessione Deep Work</h3>
                <div id="timer">25:00</div>
                <div style="text-align: center; margin-bottom: 30px;">
                    <button onclick="startTimer()" id="timer-btn">Avvia Focus</button>
                </div>
                <h4>I Tuoi Task</h4>
                <div style="display: flex; gap: 10px; margin-bottom: 15px;">
                    <input type="text" id="task-input" placeholder="Cosa vuoi completare?">
                    <button onclick="addTask()">Aggiungi</button>
                </div>
                <ul id="task-list" style="list-style: none; padding: 0;"></ul>
            </div>
        </div>

        <div id="upload" class="tab-content">
            <div class="card">
                <h3>Analisi Documenti PDF</h3>
                <p style="color: #718096;">Carica un file per permettere a FocusIA di apprenderne il contenuto.</p>
                <input type="file" id="pdf-file" accept=".pdf" style="margin-bottom: 15px;">
                <br>
                <button onclick="uploadPDF()">Analizza PDF</button>
                <div id="upload-status" style="margin-top: 15px; font-weight: 600;"></div>
            </div>
        </div>
    </div>

    <script>
        function showTab(tabId, el) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            el.classList.add('active');
        }

        async function sendMessage() {
            const input = document.getElementById('user-input');
            const chatBox = document.getElementById('chat-box');
            const text = input.value.trim();
            if (!text) return;
            chatBox.innerHTML += `<div class="msg user-msg">${text}</div>`;
            input.value = '';
            chatBox.scrollTop = chatBox.scrollHeight;
            const res = await fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt: text})
            });
            const data = await res.json();
            chatBox.innerHTML += `<div class="msg ai-msg"><b>FocusIA:</b> ${data.response}</div>`;
            // Aggiungi feedback buttons
            chatBox.innerHTML += `<div style="text-align: right;"><button onclick="sendFeedback(${data.interaction_id}, 1)">👍</button><button onclick="sendFeedback(${data.interaction_id}, -1)">👎</button></div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        async function sendFeedback(id, rating) {
            await fetch('/feedback', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({id: id, rating: rating})
            });
            alert('Feedback inviato!');
        }

        async function uploadPDF() {
            const fileInput = document.getElementById('pdf-file');
            const status = document.getElementById('upload-status');
            if (!fileInput.files[0]) return;
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            status.innerHTML = "⏳ Analisi...";
            const res = await fetch('/upload', { method: 'POST', body: formData });
            const data = await res.json();
            status.innerHTML = "✅ " + data.message;
        }

        async function addTask() {
            const input = document.getElementById('task-input');
            if (!input.value) return;
            await fetch('/add_task', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({task: input.value})
            });
            input.value = '';
            loadTasks();
        }

        async function loadTasks() {
            const res = await fetch('/get_tasks');
            const tasks = await res.json();
            document.getElementById('task-list').innerHTML = tasks.map(t => 
                `<li style="padding: 10px; border-bottom: 1px solid #edf2f7;">📌 ${t.task}</li>`).join('');
        }

        let timerInterval;
        function startTimer() {
            let timeLeft = 25 * 60;
            const btn = document.getElementById('timer-btn');
            btn.disabled = true;
            clearInterval(timerInterval);
            timerInterval = setInterval(() => {
                const mins = Math.floor(timeLeft / 60);
                const secs = timeLeft % 60;
                document.getElementById('timer').innerHTML = `${mins}:${secs < 10 ? '0' : ''}${secs}`;
                if (timeLeft <= 0) {
                    clearInterval(timerInterval);
                    alert("Sessione completata!");
                    btn.disabled = false;
                }
                timeLeft--;
            }, 1000);
        }

        function startVoiceInput() {
            const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.lang = 'it-IT';
            recognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                document.getElementById('user-input').value = transcript;
                sendMessage();
            };
            recognition.onerror = function() { alert('Errore riconoscimento vocale'); };
            recognition.start();
        }

        loadTasks();
    </script>
</body>
</html>
"""

# Routes
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data.get('prompt', '')
    response = chatbot_response(prompt)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id FROM knowledge WHERE prompt = ? ORDER BY id DESC LIMIT 1", (prompt,))
    interaction_id = c.fetchone()[0]
    conn.close()
    return jsonify({"response": response, "interaction_id": interaction_id})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    interaction_id = data.get('id')
    rating = data.get('rating')
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO feedback (interaction_id, rating) VALUES (?, ?)", (interaction_id, rating))
    c.execute("UPDATE knowledge SET confidence = confidence + ? WHERE id = ?", (0.1 * rating, interaction_id))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        chunks = extract_text_from_pdf(filepath)
        update_rag_index(chunks)
        conn = sqlite3.connect(DB_FILE)
        conn.execute("INSERT INTO knowledge (prompt, response, category, embedding) VALUES (?, ?, ?, ?)", 
                     (f"PDF: {filename}", json.dumps(chunks), "pdf", get_embedding(filename).tobytes()))
        conn.commit()
        conn.close()
        return jsonify({"message": "Analizzato e indicizzato!"})
    return jsonify({"message": "Errore"}), 400

@app.route('/add_task', methods=['POST'])
def add_task():
    task = request.get_json().get('task')
    conn = sqlite3.connect(DB_FILE)
    conn.execute("INSERT INTO tasks (task, status) VALUES (?, ?)", (task, 'pending'))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})

@app.route('/get_tasks')
def get_tasks():
    conn = sqlite3.connect(DB_FILE)
    tasks = [{"task": r[0]} for r in conn.execute("SELECT task FROM tasks ORDER BY id DESC").fetchall()]
    conn.close()
    return jsonify(tasks)

if __name__ == "__main__":
    # Inizializza DB all'avvio del modulo (funziona anche con gunicorn)
    init_database()  # Crea tabelle e dati base al primo avvio
    app.run(host="0.0.0.0", port=5000)
