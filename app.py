"""
FocusIA - Ultimate Self-Evolving AI (Render Optimized Edition)
Copyright (C) 2025 Xhulio Guranjaku
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
from flask import Flask, request, render_template_string, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import pypdf
import json
import logging

# Configurazione base
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

wikipedia.set_lang("it")
DB_FILE = "focusia_brain.db"
UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
context_lock = threading.Lock()

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configura logging
logging.basicConfig(level=logging.INFO)

# Inizializzazione database
def init_database():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS knowledge 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, prompt TEXT, response TEXT, confidence REAL, 
                  category TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS tasks 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT, status TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Conoscenza base
    base_knowledge = [
        ("FocusIA", "FocusIA è un'IA evolutiva creata da Xhulio Guranjaku."),
        ("Deep Work", "Il Deep Work è la chiave per la produttività massima.")
    ]
    for p, r in base_knowledge:
        c.execute("INSERT OR IGNORE INTO knowledge (prompt, response, confidence, category) VALUES (?, ?, 1.0, 'base')", (p, r))
    
    conn.commit()
    conn.close()

# Funzioni di Ricerca
def web_search(query):
    try:
        results = list(search(query, num=3, stop=3, pause=2))
        if results:
            res = requests.get(results[0], timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            return ' '.join([p.text for p in soup.find_all('p')[:2]])[:500]
    except: return None

def wiki_search(query):
    try: return wikipedia.summary(query, sentences=2)
    except: return None

# Gestione PDF
def extract_text_from_pdf(filepath):
    text = ""
    try:
        with open(filepath, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages[:10]: # Limite pagine per RAM
                text += page.extract_text() + "\n"
    except: pass
    return text

# Generazione risposta via API (Leggera per Render)
def get_ai_response(prompt):
    # 1. Prova OpenAI
    if OPENAI_API_KEY:
        try:
            res = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "Sei FocusIA, un'IA evolutiva creata da Xhulio Guranjaku. Rispondi in modo professionale e utile."},
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=10
            )
            return res.json()['choices'][0]['message']['content']
        except: pass

    # 2. Fallback Hugging Face
    try:
        res = requests.post(
            "https://api-inference.huggingface.co/models/gpt2",
            headers={"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"},
            json={"inputs": prompt},
            timeout=10
        )
        return res.json()[0]['generated_text']
    except:
        return "Sto imparando... chiedimi di cercare sul web o carica un PDF!"

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
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            async function uploadPDF() {
                const fileInput = document.getElementById('pdf-file');
                const status = document.getElementById('upload-status');
                if (!fileInput.files[0]) return;
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                status.innerHTML = "⏳ Analisi in corso...";
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
                        alert("Sessione completata! Ottimo lavoro.");
                        btn.disabled = false;
                    }
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
    
    # Logica di ricerca autonoma
    if any(x in prompt.lower() for x in ["chi è", "cos'è", "cerca", "news"]):
        res = wiki_search(prompt) or web_search(prompt)
        if res: return jsonify({"response": res})
    
    response = get_ai_response(prompt)
    return jsonify({"response": response})

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        text = extract_text_from_pdf(filepath)
        conn = sqlite3.connect(DB_FILE)
        conn.execute("INSERT INTO knowledge (prompt, response, category) VALUES (?, ?, ?)", 
                    (f"PDF: {filename}", text[:1000], "pdf"))
        conn.commit()
        conn.close()
        return jsonify({"message": "Documento analizzato!"})
    return jsonify({"message": "Errore caricamento"}), 400

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
    init_database()
    app.run(host="0.0.0.0", port=5000)
