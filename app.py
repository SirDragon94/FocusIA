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
from flask import Flask, request, render_template_string, send_from_directory
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
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
similarity_model = SentenceTransformer('paraphrase-distilroberta-base-v2')
context_memory = deque(maxlen=500)
wikipedia.set_lang("it")
DB_FILE = "focusia_brain.db"
CODE_FILE = "focusia_ultimate_evolving.py"
app = Flask(__name__)
context_lock = threading.Lock()

# Mini-modelli aggiuntivi
mini_models = {}

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
    c.execute('''CREATE TABLE IF NOT EXISTS code_versions 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT, performance REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS web_assets 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT, content TEXT, purpose TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS distributed_results 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, result TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS mini_models 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, path TEXT, performance REAL)''')
    c.execute("INSERT OR IGNORE INTO brain_state (key, value) VALUES ('age', 0)")
    c.execute("INSERT OR IGNORE INTO brain_state (key, value) VALUES ('curiosity', 0.95)")
    c.execute("INSERT OR IGNORE INTO brain_state (key, value) VALUES ('performance', 0.5)")
    initialize_manus_knowledge(c)
    conn.commit()
    conn.close()

# Conoscenza base di Manus IA
def initialize_manus_knowledge(cursor):
    knowledge_base = [
        {"topic": "intelligenza artificiale", "content": "L'intelligenza artificiale (IA) è una branca dell'informatica che si occupa di creare sistemi in grado di simulare comportamenti intelligenti.", "confidence": 0.98},
        {"topic": "tecnologia 2025", "content": "Nel 2025, la tecnologia ha visto progressi nell'elaborazione quantistica e IA multimodale.", "confidence": 0.95},
    ]
    for item in knowledge_base:
        embedding = similarity_model.encode(item["content"], convert_to_tensor=True).cpu().numpy()
        sentiment = analyze_sentiment(item["content"])
        cursor.execute("INSERT OR IGNORE INTO knowledge (prompt, response, confidence, category, embedding, sentiment) VALUES (?, ?, ?, ?, ?, ?)", 
                       (item["topic"], item["content"], item["confidence"], "manus_base", embedding.tobytes(), sentiment))

# Stato mentale
def get_brain_state():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT key, value FROM brain_state")
    state = dict(c.fetchall())
    conn.close()
    return state

def analyze_weaknesses():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT prompt, confidence FROM knowledge WHERE confidence < 0.7 ORDER BY usage_count DESC LIMIT 5")
    weak_entries = c.fetchall()
    conn.close()
    if not weak_entries:
        return "Ottimizza la velocità di risposta e la scalabilità."
    return f"Migliora la confidenza per: {', '.join([e[0] for e in weak_entries])}"

# Salva interazioni

def save_interaction(prompt, response, confidence, category, sentiment, embedding):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO knowledge (prompt, response, confidence, category, embedding, sentiment) VALUES (?, ?, ?, ?, ?, ?)", 
              (prompt, response, confidence, category, embedding.tobytes(), sentiment))
    conn.commit()
    conn.close()

# Generazione risposta
def generate_model_response(prompt, context=""):
    state = get_brain_state()
    input_text = context + "\nUtente: " + prompt + tokenizer.eos_token
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(
        input_ids,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=30,
        top_p=0.9,
        temperature=0.9 - (state["age"] * 0.02)
    )
    base_response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    return base_response

def enhance_response(prompt):
    base_response = generate_model_response(prompt, "\n".join(context_memory))
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT response, confidence FROM knowledge WHERE prompt LIKE ? AND confidence > 0.9 LIMIT 1", (f"%{prompt}%",))
    best_match = c.fetchone()
    conn.close()
    if best_match and random.random() < 0.3:
        return best_match[0] + " (Aggiornato online)"
    return base_response

# Valutazione qualità
def evaluate_confidence(prompt, response):
    embedding_prompt = similarity_model.encode(prompt, convert_to_tensor=True)
    embedding_response = similarity_model.encode(response, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding_prompt, embedding_response).item()

def evaluate_response_quality(prompt, response):
    conf = evaluate_confidence(prompt, response)
    length_score = min(len(response.split()) / 50, 1.0)
    sent = analyze_sentiment(response)
    return (conf * 0.5) + (length_score * 0.3) + (sent * 0.2)

def analyze_sentiment(response):
    positive = ["bene", "ottimo", "felice"]
    negative = ["male", "brutto", "triste"]
    score = 0
    for word in positive:
        if word in response.lower():
            score += 0.2
    for word in negative:
        if word in response.lower():
            score -= 0.2
    return max(-1, min(1, score))

# Risposta del chatbot
def chatbot_response(prompt):
    global context_memory
    context = "\n".join(context_memory) if context_memory else ""
    try:
        response = enhance_response(prompt)
        confidence = evaluate_confidence(prompt, response)
        sentiment = analyze_sentiment(response)
        embedding = similarity_model.encode(prompt, convert_to_tensor=True)
        save_interaction(prompt, response, confidence, "general", sentiment, embedding)
        with context_lock:
            context_memory.append(f"Utente: {prompt}\nFocusIA: {response}")
        return response, time.time()
    except Exception as e:
        logging.error(f"Errore in chatbot_response: {str(e)}")
        return "Mi dispiace, c'è stato un errore.", time.time()

# Thread di auto-evoluzione
def self_evolution_loop():
    tasks = [learn_from_advanced_ai, self_modify_with_browser, enhance_model]
    
    def run_tasks():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_task = {executor.submit(task): task for task in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Errore in {task.__name__}: {str(e)}")
    
    schedule.every(5).minutes.do(run_tasks)
    while True:
        schedule.run_pending()
        time.sleep(1)

# Route Flask
@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head><title>FocusIA</title><style>textarea, #response { font-family: Arial; }</style></head>
    <body>
        <h1>FocusIA - AI Ultra Evolutiva</h1>
        <textarea id="prompt" rows="4" cols="50" placeholder="Inserisci il tuo messaggio..."></textarea><br>
        <button onclick="sendPrompt()">Invia</button>
        <div id="response"></div>
        <div id="history"></div>
        <script>
        let history = [];
        async function sendPrompt() {
            const startTime = Date.now();
            const prompt = document.getElementById('prompt').value;
            if (!prompt.trim()) return;
            document.getElementById('response').innerHTML = 'Elaborazione...';
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt: prompt})
            });
            const data = await response.json();
            document.getElementById('response').innerHTML = `<b>Risposta:</b> ${data.response}`;
            history.push(`<b>Tu:</b> ${prompt}<br><b>FocusIA:</b> ${data.response}`);
            document.getElementById('history').innerHTML = history.join('<hr>');
            const interactionTime = (Date.now() - startTime) / 1000;
            fetch('/feedback', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt: prompt, time: interactionTime})
            });
            document.getElementById('prompt').value = '';
        }
        document.getElementById('prompt').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendPrompt();
            }
        });
        </script>
    </body>
    </html>
    """)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Prompt mancante"}), 400
    prompt = data['prompt'].strip()[:500]
    if not prompt:
        return jsonify({"error": "Prompt vuoto"}), 400
    response, _ = chatbot_response(prompt)
    return jsonify({"response": response})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    if not data or 'prompt' not in data or 'time' not in data:
        return jsonify({"error": "Dati mancanti"}), 400
    prompt = data['prompt'][:500]
    interaction_time = min(max(float(data['time']), 0), 60)
    confidence_boost = max(0, min(0.15, 1 / (interaction_time + 1)))
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("UPDATE knowledge SET confidence = MIN(1, confidence + ?) WHERE prompt = ?", 
                  (confidence_boost, prompt))
        c.execute("UPDATE brain_state SET value = value + ? WHERE key = 'performance'", 
                  (confidence_boost / 2))
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Errore database in feedback: {str(e)}")
        return jsonify({"error": "Errore interno"}), 500
    finally:
        conn.close()
    return jsonify({"status": "ok"})

@app.route('/submit_task', methods=['POST'])
def submit_task():
    data = request.get_json()
    task_id = data.get('task_id', '')
    result = data.get('result', '')
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO distributed_results (task_id, result) VALUES (?, ?)", (task_id, result))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})

@app.route('/<path:path>')
def serve_asset(path):
    return send_from_directory('static', path)

# Funzioni placeholder
def learn_from_advanced_ai():
    pass

def self_modify_with_browser():
    pass

def enhance_model():
    pass

if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")
    try:
        init_database()
        threading.Thread(target=self_evolution_loop, daemon=True).start()
        print(f"FocusIA avviato su http://localhost:5000")
        app.run(host="0.0.0.0", port=5000, threaded=True)
    except Exception as e:
        logging.critical(f"Errore avvio: {str(e)}")
        raise
