"""
FocusIA - Optimized Self-Evolving AI for Render Free Tier with Supabase
Copyright (C) 2025 Xhulio Guranjaku
"""

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
import faiss
import psutil
from supabase import create_client, Client

# Configurazione base
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

wikipedia.set_lang("it")
UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
context_memory = deque(maxlen=10)
context_lock = threading.Lock()
index = None
knowledge_chunks = []
embedding_cache = {}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

logging.basicConfig(level=logging.INFO)

# Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

if not supabase:
    logging.error("SUPABASE_URL o SUPABASE_KEY mancanti nelle env vars!")

# Embedding via HF API
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
    except Exception as e:
        logging.error(f"Errore embedding: {e}")
        return np.zeros(384)

# Update RAG index
def update_rag_index(new_chunks):
    global index, knowledge_chunks
    if index is None:
        index = faiss.IndexFlatL2(384)
    embeddings = np.array([get_embedding(chunk) for chunk in new_chunks])
    index.add(embeddings)
    knowledge_chunks.extend(new_chunks)

# Retrieve RAG
def retrieve_relevant_chunks(query, top_k=2):
    if index is None or not hasattr(index, 'ntotal') or index.ntotal == 0:
        return []
    try:
        query_emb = get_embedding(query)
        D, I = index.search(np.array([query_emb]), top_k)
        return [knowledge_chunks[i] for i in I[0] if i < len(knowledge_chunks)]
    except Exception as e:
        logging.error(f"Errore retrieval RAG: {e}")
        return []

# Save interaction in knowledge
def save_interaction(prompt, response, confidence, category, sentiment, embedding):
    if not supabase:
        return
    data = {
        "prompt": prompt,
        "response": response,
        "confidence": confidence,
        "category": category,
        "usage_count": 0,
        "sentiment": sentiment,
        "embedding": json.dumps(embedding.tolist()) if embedding is not None and embedding.any() else None
    }
    try:
        supabase.table("knowledge").insert(data).execute()
    except Exception as e:
        logging.error(f"Errore save_interaction: {e}")

# Update brain_state
def update_brain_state(age_inc=0.1, curiosity_adj=0, empathy_adj=0, clusters_adj=0):
    if not supabase:
        return
    try:
        # Age
        age_res = supabase.table("brain_state").select("value").eq("key", "age").execute()
        age_val = age_res.data[0]["value"] if age_res.data else 0.0
        supabase.table("brain_state").upsert({"key": "age", "value": age_val + age_inc}).execute()

        # Curiosity
        if curiosity_adj != 0:
            cur_res = supabase.table("brain_state").select("value").eq("key", "curiosity").execute()
            cur_val = cur_res.data[0]["value"] if cur_res.data else 0.8
            supabase.table("brain_state").update({"value": max(0, min(1, cur_val + curiosity_adj))}).eq("key", "curiosity").execute()

        # Empathy
        if empathy_adj != 0:
            emp_res = supabase.table("brain_state").select("value").eq("key", "empathy").execute()
            emp_val = emp_res.data[0]["value"] if emp_res.data else 0.5
            supabase.table("brain_state").update({"value": max(0, min(1, emp_val + empathy_adj))}).eq("key", "empathy").execute()

        # Clusters
        if clusters_adj != 0:
            clu_res = supabase.table("brain_state").select("value").eq("key", "knowledge_clusters").execute()
            clu_val = clu_res.data[0]["value"] if clu_res.data else 3.0
            supabase.table("brain_state").update({"value": clu_val + clusters_adj}).eq("key", "knowledge_clusters").execute()
    except Exception as e:
        logging.error(f"Errore update_brain_state: {str(e)}")

# Get brain_state
def get_brain_state():
    if not supabase:
        return {"age": 0.0, "curiosity": 0.8, "empathy": 0.5, "knowledge_clusters": 3.0}
    try:
        result = supabase.table("brain_state").select("*").execute()
        return {row["key"]: row["value"] for row in result.data}
    except:
        return {"age": 0.0, "curiosity": 0.8, "empathy": 0.5, "knowledge_clusters": 3.0}

# Search database (simplified similarity)
def search_database(prompt):
    if not supabase:
        return None
    try:
        result = supabase.table("knowledge").select("prompt, response, confidence, usage_count, id, embedding, cluster").execute()
        interactions = result.data
        if not interactions:
            return None
        prompt_emb = get_embedding(prompt)
        best_match = None
        best_score = -1
        for item in interactions:
            past_emb_str = item["embedding"]
if past_emb_str and past_emb_str.strip():
    try:
        past_emb = np.array(json.loads(past_emb_str))
    except json.JSONDecodeError:
        past_emb = np.zeros(384)
else:
    past_emb = np.zeros(384)
            similarity = np.dot(prompt_emb, past_emb) / (np.linalg.norm(prompt_emb) * np.linalg.norm(past_emb) + 1e-8)
            score = (similarity * 0.7) + (item["confidence"] * 0.2) + (min(item["usage_count"], 10) * 0.1 / 10)
            if score > best_score and similarity > 0.6:
                best_score = score
                best_match = (item["response"], item["id"])
        if best_match:
            supabase.table("knowledge").update({"usage_count": supabase.table("knowledge").select("usage_count").eq("id", best_match[1]).execute().data[0]["usage_count"] + 1}).eq("id", best_match[1]).execute()
            return best_match[0]
        return None
    except Exception as e:
        logging.error(f"Errore search_database: {e}")
        return None

# Evolve prompts
def evolve_prompts():
    if not supabase:
        return
    try:
        # Conteggio corretto con Supabase (usa count=True e prendi result.count)
        count_result = supabase.table("knowledge").select("*", count="exact").execute()
        interaction_count = count_result.count or 0
        if interaction_count % 5 != 0:
            return
        # Resto invariato...
        best_prompt_result = supabase.table("prompts").select("system_prompt").order("score", desc=True).limit(1).execute()
        best_prompt = best_prompt_result.data[0]["system_prompt"] if best_prompt_result.data else "Sei FocusIA, un'IA evolutiva."
        # ... continua con variants, test, ecc.
    except Exception as e:
        logging.error(f"Errore evolve_prompts: {e}")

# AI response
def get_ai_response(prompt, system_prompt=None):
    evolve_prompts()
    if not supabase:
        system_prompt = "Sei FocusIA, un'IA evolutiva creata da Xhulio Guranjaku. Rispondi in modo professionale e utile."
    else:
        result = supabase.table("prompts").select("system_prompt").order("score", desc=True).limit(1).execute()
        system_prompt = result.data[0]["system_prompt"] if result.data else "Sei FocusIA..."

    relevant = retrieve_relevant_chunks(prompt)
    augmented_prompt = prompt + "\nRilevante: " + " ".join(relevant)

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
                timeout=15
            )
            return res.json()['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"Errore OpenAI: {e}")
    try:
        res = requests.post(
            "https://api-inference.huggingface.co/models/distilgpt2",  # Modello cambiato per stabilità
            headers={"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"},
            json={"inputs": augmented_prompt},
            timeout=30
        )
        logging.info(f"HF status: {res.status_code}, text: {res.text[:200]}")
        return res.json()[0]['generated_text']
    except Exception as e:
        logging.error(f"Errore HF fallback dettagliato: {str(e)}")
        return "Sto imparando... chiedimi di cercare o carica un PDF!"

# Web search
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

# Curiosity learning
def curiosity_learning():
    state = get_brain_state()
    if random.random() < state.get("curiosity", 0.8):
        try:
            result = supabase.table("knowledge").select("prompt").lt("confidence", 0.7).order("timestamp", desc=True).limit(1).execute()
            weak = result.data
            if weak:
                weak_prompt = weak[0]["prompt"]
                new_res = web_search_multi(weak_prompt)
                if new_res:
                    conf = random.uniform(0.6, 0.9)
                    sent = 0.0
                    emb = get_embedding(weak_prompt)
                    save_interaction(weak_prompt, new_res, conf, "curiosity", sent, emb)
                    update_brain_state(curiosity_adj=-0.05 if conf > 0.7 else 0.05)
        except Exception as e:
            logging.error(f"Errore curiosity: {e}")

# Sentiment analysis
def analyze_sentiment(response):
    positive = ["bene", "ottimo", "felice"]
    negative = ["male", "brutto", "triste"]
    score = sum(1 for w in positive if w in response.lower()) - sum(1 for w in negative if w in response.lower())
    return max(-1, min(1, score * 0.2))

# PDF extract
def extract_text_from_pdf(filepath):
    text = ""
    with open(filepath, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages[:5]:
            text += page.extract_text() + "\n"
    chunks = [text[i:i+200] for i in range(0, len(text), 200)]
    return chunks

# Chatbot response
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
        conf = random.uniform(0.7, 0.95)
        sent = analyze_sentiment(response)
        emb = get_embedding(prompt)
        save_interaction(prompt, response, conf, "general", sent, emb)
        update_brain_state()
    personality = random.choice([
        " Con focus...",
        f" Curiosità: {state.get('curiosity', 0.8):.1f}..."
    ])
    response += personality
    context_memory.append(f"Utente: {prompt}\nFocusIA: {response}")
    threading.Thread(target=curiosity_learning).start()
    return response

HTML = """
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FocusIA - Dashboard Evolutivo</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5; margin: 0; display: flex; height: 100vh; }
        #sidebar { width: 260px; background: #1a202c; color: white; padding: 25px; display: flex; flex-direction: column; }
        #main { flex: 1; padding: 30px; overflow-y: auto; background: #ffffff; }
        .card { background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 25px; border: 1px solid #e2e8f0; }
        h2, h3 { color: #2d3748; margin-top: 0; }
        #chat-box { height: 400px; overflow-y: auto; border: 1px solid #edf2f7; padding: 15px; margin-bottom: 15px; background: #f8fafc; border-radius: 8px; display: flex; flex-direction: column; }
        .msg { margin-bottom: 12px; padding: 10px 15px; border-radius: 8px; max-width: 85%; word-wrap: break-word; }
        .user-msg { background: #ebf8ff; color: #2b6cb0; align-self: flex-end; font-weight: 600; }
        .ai-msg { background: #f0fff4; color: #2f855a; border: 1px solid #c6f6d5; align-self: flex-start; }
        input[type="text"] { width: calc(100% - 140px); padding: 12px; border: 1px solid #cbd5e0; border-radius: 6px; outline: none; }
        button { background: #4299e1; color: white; border: none; padding: 12px 20px; border-radius: 6px; cursor: pointer; font-weight: 600; transition: 0.2s; margin-left: 10px; }
        button:hover { background: #3182ce; }
        .nav-item { cursor: pointer; padding: 12px; border-radius: 8px; margin-bottom: 8px; transition: 0.2s; }
        .nav-item:hover { background: #2d3748; }
        .nav-item.active { background: #4299e1; }
        .tab-content { display: none; animation: fadeIn 0.3s; }
        .tab-content.active { display: block; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        #timer { font-size: 48px; font-weight: bold; color: #e53e3e; text-align: center; margin: 20px 0; }
        #feedback-buttons { margin-top: 5px; text-align: right; }
        #feedback-buttons button { padding: 6px 12px; font-size: 16px; margin-left: 5px; }
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
        <!-- Tab Chat -->
        <div id="chat" class="tab-content active">
            <div class="card">
                <h3>Assistente Evolutivo</h3>
                <div id="chat-box"></div>
                <div style="display: flex; gap: 10px; margin-top: 15px;">
                    <input type="text" id="user-input" placeholder="Chiedi qualcosa o scrivi 'cerca...'">
                    <button onclick="sendMessage()">Invia</button>
                    <button onclick="startVoiceInput()">🎤 Parla</button>
                </div>
            </div>
        </div>

        <!-- Tab Deep Work -->
        <div id="tasks" class="tab-content">
            <div class="card">
                <h3>Sessione Deep Work</h3>
                <div id="timer">25:00</div>
                <div style="text-align: center; margin: 20px 0;">
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

        <!-- Tab Upload -->
        <div id="upload" class="tab-content">
            <div class="card">
                <h3>Analisi Documenti PDF</h3>
                <p style="color: #718096;">Carica un file per permettere a FocusIA di apprenderne il contenuto.</p>
                <input type="file" id="pdf-file" accept=".pdf" style="margin-bottom: 15px;">
                <button onclick="uploadPDF()">Analizza PDF</button>
                <div id="upload-status" style="margin-top: 15px; font-weight: 600; color: #2f855a;"></div>
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

            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt: text})
                });
                if (!res.ok) throw new Error('Errore server');
                const data = await res.json();

                chatBox.innerHTML += `<div class="msg ai-msg"><b>FocusIA:</b> ${data.response}</div>`;
                if (data.interaction_id) {
                    chatBox.innerHTML += `
                        <div id="feedback-buttons">
                            <button onclick="sendFeedback(${data.interaction_id}, 1)">👍</button>
                            <button onclick="sendFeedback(${data.interaction_id}, -1)">👎</button>
                        </div>`;
                }
                chatBox.scrollTop = chatBox.scrollHeight;
            } catch (err) {
                chatBox.innerHTML += `<div class="msg ai-msg" style="color:red;">Errore: ${err.message}</div>`;
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        }

        async function sendFeedback(id, rating) {
            await fetch('/feedback', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({id: id, rating: rating})
            });
            alert('Grazie per il feedback!');
        }

        async function uploadPDF() {
            const fileInput = document.getElementById('pdf-file');
            const status = document.getElementById('upload-status');
            if (!fileInput.files[0]) return;
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            status.innerHTML = "⏳ Analisi in corso...";
            try {
                const res = await fetch('/upload', { method: 'POST', body: formData });
                const data = await res.json();
                status.innerHTML = "✅ " + data.message;
            } catch (err) {
                status.innerHTML = "❌ Errore: " + err.message;
            }
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

        // Carica task all'apertura
        loadTasks();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data.get('prompt', '')
    logging.info(f"Ricevuto prompt: {prompt}")
    try:
        response = chatbot_response(prompt)
        result = supabase.table("knowledge").select("id").order("timestamp", desc=True).limit(1).execute()
        interaction_id = result.data[0]["id"] if result.data else None
        logging.info(f"Risposta generata: {response[:100]}...")
        return jsonify({"response": response, "interaction_id": interaction_id})
    except Exception as e:
        logging.error(f"Errore /chat: {str(e)}")
        return jsonify({"response": "Errore interno, riprova.", "interaction_id": None}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    interaction_id = data.get('id')
    rating = data.get('rating')
    if not supabase or not interaction_id:
        return jsonify({"status": "ok"})
    try:
        supabase.table("feedback").insert({"interaction_id": interaction_id, "rating": rating}).execute()
        current_conf = supabase.table("knowledge").select("confidence").eq("id", interaction_id).execute().data[0]["confidence"]
        new_conf = current_conf + 0.1 * rating
        supabase.table("knowledge").update({"confidence": new_conf}).eq("id", interaction_id).execute()
    except Exception as e:
        logging.error(f"Errore feedback: {e}")
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
        if supabase:
            data = {
                "prompt": f"PDF: {filename}",
                "response": json.dumps(chunks),
                "category": "pdf",
                "embedding": get_embedding(filename).tolist()
            }
            supabase.table("knowledge").insert(data).execute()
        return jsonify({"message": "Analizzato e indicizzato!"})
    return jsonify({"message": "Errore"}), 400

@app.route('/add_task', methods=['POST'])
def add_task():
    task = request.get_json().get('task')
    if supabase:
        supabase.table("tasks").insert({"task": task, "status": "pending"}).execute()
    return jsonify({"status": "ok"})

@app.route('/get_tasks')
def get_tasks():
    if not supabase:
        return jsonify([])
    try:
        result = supabase.table("tasks").select("task").order("id", desc=True).execute()
        tasks = [{"task": row["task"]} for row in result.data]
        return jsonify(tasks)
    except:
        return jsonify([])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
