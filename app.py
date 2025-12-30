"""
FocusIA - Ultimate Self-Evolving AI
Copyright (C) 2025 Xhulio Guranjaku
"""

import sqlite3
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer, util
import random
from collections import deque
import os
import threading
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import wikipedia
import time
import numpy as np
from flask import Flask, request, render_template_string, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import pypdf
import ast
import inspect
import json
import concurrent.futures
import hashlib
import logging
import schedule

# Configurazione base
model_name = "sshleifer/tiny-gpt2"
token = os.getenv("HUGGINGFACE_TOKEN")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
except Exception as e:
    logging.error(f"Errore caricamento modello: {e}")
    # Fallback o gestione errore
    tokenizer = None
    model = None

similarity_model = SentenceTransformer('paraphrase-distilroberta-base-v2')
context_memory = deque(maxlen=500)
wikipedia.set_lang("it")
DB_FILE = "focusia_brain.db"
UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
context_lock = threading.Lock()

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configura logging
logging.basicConfig(level=logging.INFO, filename="focusia.log")

# Inizializzazione database
def init_database():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS knowledge 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, prompt TEXT, response TEXT, confidence REAL, 
                  usage_count INTEGER DEFAULT 0, category TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
                  embedding BLOB, sentiment REAL, cluster INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS brain_state 
                 (key TEXT PRIMARY KEY, value REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS tasks 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT, priority INTEGER, status TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute("INSERT OR IGNORE INTO brain_state (key, value) VALUES ('age', 0)")
    c.execute("INSERT OR IGNORE INTO brain_state (key, value) VALUES ('curiosity', 0.95)")
    c.execute("INSERT OR IGNORE INTO brain_state (key, value) VALUES ('performance', 0.5)")
    initialize_manus_knowledge(c)
    conn.commit()
    conn.close()

def initialize_manus_knowledge(cursor):
    knowledge_base = [
        {"topic": "FocusIA", "content": "FocusIA è un'IA auto-evolutiva creata da Xhulio Guranjaku per assistere gli utenti con ricerca, analisi e gestione del tempo.", "confidence": 1.0},
        {"topic": "Deep Work", "content": "Il Deep Work è la capacità di concentrarsi senza distrazioni su un compito cognitivamente impegnativo.", "confidence": 0.95},
    ]
    for item in knowledge_base:
        embedding = similarity_model.encode(item["content"], convert_to_tensor=True).cpu().numpy()
        cursor.execute("INSERT OR IGNORE INTO knowledge (prompt, response, confidence, category, embedding, sentiment) VALUES (?, ?, ?, ?, ?, ?)", 
                       (item["topic"], item["content"], item["confidence"], "base", embedding.tobytes(), 0.5))

# Funzioni di Ricerca
def web_search(query):
    try:
        results = []
        for j in search(query, num=3, stop=3, pause=2):
            results.append(j)
        if results:
            res = requests.get(results[0], timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            text = ' '.join([p.text for p in soup.find_all('p')[:3]])
            return text[:500]
    except Exception as e:
        logging.error(f"Errore ricerca web: {e}")
    return None

def wiki_search(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except:
        return None

# Gestione PDF
def extract_text_from_pdf(filepath):
    text = ""
    try:
        with open(filepath, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        logging.error(f"Errore estrazione PDF: {e}")
    return text

# Generazione risposta
def chatbot_response(prompt):
    global context_memory
    
    # 1. Cerca nel database locale
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT response FROM knowledge WHERE prompt LIKE ? AND confidence > 0.8 ORDER BY confidence DESC LIMIT 1", (f"%{prompt}%",))
    row = c.fetchone()
    conn.close()
    
    if row:
        return row[0], "database"

    # 2. Cerca su Wikipedia/Web se necessario
    if any(word in prompt.lower() for word in ["chi è", "cos'è", "storia", "ricerca"]):
        wiki_res = wiki_search(prompt)
        if wiki_res:
            return wiki_res, "wikipedia"
        web_res = web_search(prompt)
        if web_res:
            return web_res, "web"

    # 3. Genera con il modello
    if model and tokenizer:
        input_text = "\n".join(context_memory) + "\nUtente: " + prompt + tokenizer.eos_token
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        output_ids = model.generate(input_ids, max_length=150, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.7)
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    else:
        response = "Sto imparando... chiedimi qualcos'altro o carica un documento!"

    # Salva in memoria
    with context_lock:
        context_memory.append(f"Utente: {prompt}\nFocusIA: {response}")
    
    return response, "model"

# Route Flask
@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="it">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FocusIA - Dashboard</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f7f6; margin: 0; display: flex; }
            #sidebar { width: 250px; background: #2c3e50; color: white; height: 100vh; padding: 20px; }
            #main { flex: 1; padding: 20px; overflow-y: auto; }
            .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
            #chat-box { height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; background: #fff; }
            .user-msg { color: #2980b9; font-weight: bold; }
            .ai-msg { color: #27ae60; margin-bottom: 10px; }
            input[type="text"], textarea { width: 100%; padding: 10px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 4px; }
            button { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
            button:hover { background: #2980b9; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            .nav-item { cursor: pointer; padding: 10px; border-bottom: 1px solid #34495e; }
            .nav-item:hover { background: #34495e; }
        </style>
    </head>
    <body>
        <div id="sidebar">
            <h2>FocusIA</h2>
            <div class="nav-item" onclick="showTab('chat')">Chat AI</div>
            <div class="nav-item" onclick="showTab('tasks')">Gestione Tempo</div>
            <div class="nav-item" onclick="showTab('upload')">Carica Documenti</div>
        </div>
        <div id="main">
            <div id="chat" class="tab-content active">
                <div class="card">
                    <h3>Chat Evolutiva</h3>
                    <div id="chat-box"></div>
                    <input type="text" id="user-input" placeholder="Scrivi un messaggio...">
                    <button onclick="sendMessage()">Invia</button>
                </div>
            </div>
            
            <div id="tasks" class="tab-content">
                <div class="card">
                    <h3>Deep Work & Task</h3>
                    <input type="text" id="task-input" placeholder="Nuovo task...">
                    <button onclick="addTask()">Aggiungi Task</button>
                    <ul id="task-list"></ul>
                    <hr>
                    <h4>Timer Deep Work</h4>
                    <div id="timer">25:00</div>
                    <button onclick="startTimer()">Avvia Sessione</button>
                </div>
            </div>

            <div id="upload" class="tab-content">
                <div class="card">
                    <h3>Analisi PDF</h3>
                    <input type="file" id="pdf-file" accept=".pdf">
                    <button onclick="uploadPDF()">Carica e Analizza</button>
                    <div id="upload-status"></div>
                </div>
            </div>
        </div>

        <script>
            function showTab(tabId) {
                document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
                document.getElementById(tabId).classList.add('active');
            }

            async function sendMessage() {
                const input = document.getElementById('user-input');
                const chatBox = document.getElementById('chat-box');
                const text = input.value;
                if (!text) return;

                chatBox.innerHTML += `<div class="user-msg">Tu: ${text}</div>`;
                input.value = '';

                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt: text})
                });
                const data = await res.json();
                chatBox.innerHTML += `<div class="ai-msg">FocusIA: ${data.response}</div>`;
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            async function uploadPDF() {
                const fileInput = document.getElementById('pdf-file');
                const status = document.getElementById('upload-status');
                if (!fileInput.files[0]) return;

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                status.innerHTML = "Caricamento in corso...";
                const res = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                status.innerHTML = data.message;
            }

            async function addTask() {
                const input = document.getElementById('task-input');
                const text = input.value;
                if (!text) return;
                await fetch('/add_task', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({task: text})
                });
                input.value = '';
                loadTasks();
            }

            async function loadTasks() {
                const res = await fetch('/get_tasks');
                const tasks = await res.json();
                const list = document.getElementById('task-list');
                list.innerHTML = tasks.map(t => `<li>${t.task} [${t.status}]</li>`).join('');
            }

            let timerInterval;
            function startTimer() {
                let timeLeft = 25 * 60;
                clearInterval(timerInterval);
                timerInterval = setInterval(() => {
                    const mins = Math.floor(timeLeft / 60);
                    const secs = timeLeft % 60;
                    document.getElementById('timer').innerHTML = `${mins}:${secs < 10 ? '0' : ''}${secs}`;
                    if (timeLeft <= 0) clearInterval(timerInterval);
                    timeLeft--;
                }, 1000);
            }
            
            loadTasks();
        </script>
    </body>
    </html>
    """)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data.get('prompt', '')
    response, source = chatbot_response(prompt)
    return jsonify({"response": response, "source": source})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"message": "Nessun file"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "Nessun file selezionato"}), 400
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Estrai testo e salva in knowledge
        text = extract_text_from_pdf(filepath)
        if text:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            embedding = similarity_model.encode(text[:500], convert_to_tensor=True).cpu().numpy()
            c.execute("INSERT INTO knowledge (prompt, response, confidence, category, embedding) VALUES (?, ?, ?, ?, ?)", 
                      (f"Contenuto di {filename}", text[:2000], 1.0, "pdf_upload", embedding.tobytes()))
            conn.commit()
            conn.close()
            return jsonify({"message": f"File {filename} analizzato con successo!"})
    return jsonify({"message": "Formato non supportato"}), 400

@app.route('/add_task', methods=['POST'])
def add_task():
    data = request.get_json()
    task = data.get('task', '')
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO tasks (task, priority, status) VALUES (?, ?, ?)", (task, 1, 'pending'))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})

@app.route('/get_tasks')
def get_tasks():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT task, status FROM tasks ORDER BY timestamp DESC")
    tasks = [{"task": row[0], "status": row[1]} for row in c.fetchall()]
    conn.close()
    return jsonify(tasks)

# Thread di auto-evoluzione (placeholder implementati)
def self_evolution_loop():
    while True:
        # Simula apprendimento
        logging.info("FocusIA sta evolvendo...")
        time.sleep(300)

if __name__ == "__main__":
    init_database()
    threading.Thread(target=self_evolution_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
