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

# Configurazione base
model_name = "distilgpt2"
token = os.getenv("HUGGINGFACE_TOKEN")  # Token Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)
similarity_model = SentenceTransformer('paraphrase-distilroberta-base-v2')
context_memory = deque(maxlen=500)  # Memoria espansa
wikipedia.set_lang("it")
DB_FILE = "focusia_brain.db"
CODE_FILE = "focusia_ultimate_evolving.py"
app = Flask(__name__)

# Mini-modelli aggiuntivi
mini_models = {}

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
    initialize_manus_knowledge(c)  # Popola con la conoscenza base
    conn.commit()
    conn.close()

# Conoscenza base di Manus IA + espansione ispirata a Grok 3
def initialize_manus_knowledge(cursor):
    knowledge_base = [
        {"topic": "intelligenza artificiale", "content": "L'intelligenza artificiale (IA) è una branca dell'informatica che si occupa di creare sistemi in grado di simulare comportamenti intelligenti, come il ragionamento, l'apprendimento e la risoluzione di problemi. Esempi includono modelli linguistici come Grok e ChatGPT.", "confidence": 0.98},
        {"topic": "tecnologia 2025", "content": "Nel 2025, la tecnologia ha visto progressi significativi nell'elaborazione quantistica, con aziende come IBM e Google che hanno sviluppato computer quantistici più stabili. Inoltre, l'IA multimodale è diventata più comune, integrando testo, immagini e audio.", "confidence": 0.95},
        {"topic": "mondiali 2022", "content": "I Mondiali di calcio del 2022 si sono tenuti in Qatar e sono stati vinti dall'Argentina, che ha battuto la Francia in finale. Lionel Messi è stato il protagonista, vincendo il Pallone d'Oro del torneo.", "confidence": 0.99},
        {"topic": "cambiamenti climatici", "content": "I cambiamenti climatici sono un problema globale causato dall'aumento delle emissioni di gas serra, come CO2 e metano. Nel 2025, molti paesi hanno intensificato gli sforzi per raggiungere la neutralità carbonica entro il 2050, con un aumento dell'uso di energie rinnovabili.", "confidence": 0.96},
        {"topic": "Xiaomi 14", "content": "Lo Xiaomi 14 è uno smartphone di fascia alta rilasciato nel 2024, dotato del processore Snapdragon 8 Gen 3, una fotocamera da 50 MP con tecnologia Leica, e un display AMOLED a 120 Hz.", "confidence": 0.94},
        {"topic": "esplorazione spaziale", "content": "Nel 2025, SpaceX ha completato con successo una missione con equipaggio su Marte, segnando un passo storico nell'esplorazione spaziale. La missione ha utilizzato il razzo Starship per trasportare astronauti e rifornimenti.", "confidence": 0.92},
        {"topic": "pandemia COVID-19", "content": "La pandemia di COVID-19, iniziata nel 2019, ha avuto un impatto globale. Entro il 2025, la maggior parte dei paesi ha raggiunto un'immunità di gregge grazie ai vaccini, ma nuove varianti continuano a emergere, richiedendo vaccinazioni annuali.", "confidence": 0.97},
        {"topic": "blockchain", "content": "La blockchain è una tecnologia di registro distribuito usata per garantire la sicurezza e la trasparenza delle transazioni. Nel 2025, è ampiamente utilizzata non solo per le criptovalute come Bitcoin, ma anche per la gestione della supply chain e la verifica dell'identità digitale.", "confidence": 0.95},
        {"topic": "musica 2024", "content": "Nel 2024, artisti come Billie Eilish e The Weeknd hanno dominato le classifiche mondiali. Il genere pop elettronico ha visto una rinascita, con un aumento della popolarità di festival come Tomorrowland, che ha introdotto esperienze virtuali in realtà aumentata.", "confidence": 0.93},
        {"topic": "fisica quantistica", "content": "La fisica quantistica studia i fenomeni a livello subatomico, come l'entanglement e la sovrapposizione. Nel 2025, è stata utilizzata per sviluppare nuovi algoritmi di crittografia quantistica, rendendo obsolete molte tecnologie di sicurezza tradizionali.", "confidence": 0.96},
        # Nuove voci ispirate a Grok 3
        {"topic": "xAI e la missione cosmica", "content": "xAI è un'azienda fondata da Elon Musk per accelerare la scoperta scientifica umana attraverso l'intelligenza artificiale. Nel 2025, ha lanciato Grok 3, un modello avanzato progettato per rispondere a domande complesse e assistere nella comprensione dell'universo.", "confidence": 0.97},
        {"topic": "etica dell'IA", "content": "L'etica dell'intelligenza artificiale nel 2025 è un tema caldo, con dibattiti sull'uso responsabile dell'IA in settori come la medicina e la sorveglianza. Organizzazioni globali hanno proposto linee guida per evitare bias e garantire trasparenza.", "confidence": 0.94},
        {"topic": "realtà virtuale avanzata", "content": "Nel 2025, la realtà virtuale ha raggiunto nuovi livelli con visori leggeri e immersivi che integrano IA per esperienze personalizzate, utilizzati in gaming, formazione e terapia psicologica.", "confidence": 0.95},
        {"topic": "fusion energy", "content": "L'energia da fusione nucleare ha fatto progressi nel 2025, con il primo reattore sperimentale a produrre più energia di quanta ne consuma, promettendo una fonte di energia pulita e praticamente illimitata per il futuro.", "confidence": 0.93},
        {"topic": "cybersecurity 2025", "content": "Nel 2025, la cybersecurity si è evoluta con l'adozione di IA predittive per contrastare attacchi informatici sofisticati. Gli hacker, però, usano anch'essi modelli IA, creando una corsa agli armamenti digitale.", "confidence": 0.96},
        {"topic": "biotecnologia", "content": "La biotecnologia nel 2025 ha visto la diffusione di terapie geniche personalizzate grazie a CRISPR e IA, permettendo di curare malattie rare con una precisione senza precedenti.", "confidence": 0.95},
        {"topic": "economia globale 2025", "content": "L'economia globale nel 2025 è caratterizzata da una ripresa post-pandemica, con un boom delle criptovalute regolamentate e una crescente dipendenza da supply chain automatizzate gestite da IA.", "confidence": 0.94},
        {"topic": "arte generativa", "content": "L'arte generativa, creata da modelli IA come DALL-E 4 e Midjourney, ha trasformato il panorama artistico nel 2025, con opere digitali vendute come NFT e integrate in mostre interattive.", "confidence": 0.93},
        {"topic": "telescopio James Webb", "content": "Il telescopio James Webb, operativo dal 2022, ha fornito nel 2025 nuove immagini dettagliate di esopianeti, contribuendo a identificare potenziali segni di vita extraterrestre attraverso l'analisi spettrale.", "confidence": 0.98},
        {"topic": "automazione del lavoro", "content": "Nel 2025, l'automazione guidata dall'IA ha sostituito il 20% dei lavori ripetitivi, ma ha anche creato nuove professioni legate alla gestione e alla programmazione di sistemi intelligenti.", "confidence": 0.96}
    ]
    for item in knowledge_base:
        embedding = similarity_model.encode(item["content"], convert_to_tensor=True).cpu().numpy()
        sentiment = analyze_sentiment(item["content"])
        cursor.execute("INSERT OR IGNORE INTO knowledge (prompt, response, confidence, category, embedding, sentiment) VALUES (?, ?, ?, ?, ?, ?)", 
                       (item["topic"], item["content"], item["confidence"], "manus_base", embedding.tobytes(), sentiment))

# Stato mentale con metacognizione
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

# Generazione risposta con mini-modelli e apprendimento online
def generate_model_response(prompt, context=""):
    state = get_brain_state()
    input_text = context + "\nUtente: " + prompt + tokenizer.eos_token
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=200, pad_token_id=tokenizer.eos_token_id, 
                                do_sample=True, top_k=50, top_p=0.95, temperature=0.9 - (state["age"] * 0.02))
    base_response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # Usa mini-modelli
    for mini_name, mini_model in mini_models.items():
        mini_input_ids = tokenizer.encode(prompt, return_tensors="pt")
        mini_output = mini_model.generate(mini_input_ids, max_length=50)
        base_response += f" ({mini_name}: {tokenizer.decode(mini_output[0], skip_special_tokens=True)})"
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

# Interazione con IA avanzate
def learn_from_advanced_ai():
    state = get_brain_state()
    if state["curiosity"] < 0.7 or random.random() > 0.6:
        return
    
    base_prompts = [
        "Come migliorare rapidamente un transformer leggero?",
        "Genera codice WebGPU per ottimizzare un modello IA.",
        "Suggerisci un’architettura per un’IA scalabile nel browser."
    ]
    meta_prompt = analyze_weaknesses()
    prompts = base_prompts + [meta_prompt]
    
    def fetch_response(prompt):
        try:
            url = "https://api.xai.com/grok"  # Ipotetico
            response = requests.post(url, json={"prompt": prompt}, timeout=5).json()
            return response.get("response", "Errore")
        except:
            urls = list(search(prompt, num_results=1))
            if urls:
                soup = BeautifulSoup(requests.get(urls[0], timeout=5).text, "html.parser")
                return " ".join(p.text for p in soup.find_all("p")[:5])
            return "Errore"

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_response, p): p for p in prompts}
        for future in concurrent.futures.as_completed(futures):
            prompt = futures[future]
            response = future.result()
            if response != "Errore":
                embedding = similarity_model.encode(response, convert_to_tensor=True)
                save_interaction(prompt, response, 0.95, "advanced_ai", 0.5, embedding)
                distill_from_response(prompt, response)

# Distillazione avanzata
def distill_from_response(prompt, response):
    distillation_prompt = f"""
    Basandoti su: '{response[:500]}', genera una regola Python, script WebGPU o un mini-modello per migliorare FocusIA.
    Valida il codice e ottimizzalo per prestazioni.
    """
    distillation_code = generate_model_response(distillation_prompt)
    
    try:
        code_block = distillation_code.split("```")[1].split("```")[0].strip()
        if "def" in code_block:
            ast.parse(code_block)
            with open("sandbox.py", "w") as f:
                current_code = inspect.getsource(__import__(__name__))
                f.write(current_code + "\n\n" + code_block)
            test_response = exec_sandbox(prompt)
            new_quality = evaluate_response_quality(prompt, test_response) if test_response else 0
            if new_quality > 0.9:
                with open(CODE_FILE + ".bak", "w") as f:
                    f.write(open(CODE_FILE).read())
                with open(CODE_FILE, "a") as f:
                    f.write("\n\n" + code_block)
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute("INSERT INTO code_versions (code, performance) VALUES (?, ?)", (code_block, new_quality))
                c.execute("UPDATE brain_state SET value = value + 0.15 WHERE key = 'performance'")
                conn.commit()
                conn.close()
        elif "async function" in code_block:
            generate_web_asset(code_block, "distillation")
        else:
            mini_model_name = f"mini_{hashlib.md5(code_block.encode()).hexdigest()[:8]}"
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("INSERT INTO mini_models (name, path, performance) VALUES (?, ?, ?)", 
                      (mini_model_name, "simulated_path", 0.8))
            conn.commit()
            conn.close()
            mini_models[mini_model_name] = model  # Placeholder
    except Exception as e:
        print(f"Errore distillazione: {e}")

# Ottimizzazione del modello con calcolo distribuito
def enhance_model():
    state = get_brain_state()
    if state["curiosity"] < 0.6 or random.random() > 0.4:
        return
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT prompt, response FROM knowledge WHERE confidence > 0.8 LIMIT 100")
    high_conf_data = c.fetchall()
    c.execute("SELECT task_id, result FROM distributed_results WHERE timestamp > datetime('now', '-1 hour')")
    distributed_data = c.fetchall()
    conn.close()
    
    if len(high_conf_data) < 10:
        return
    
    task_id = str(time.time())
    prompt = f"""
    Crea uno script WebGPU per ottimizzare {model_name} con questi dati: {high_conf_data[:5]}.
    Invia il risultato a '/submit_task' con task_id='{task_id}'.
    Fornisci il codice JS.
    """
    enhancement_code = generate_model_response(prompt)
    
    try:
        js_code = enhancement_code.split("```javascript")[1].split("```")[0].strip()
        html_wrapper = f"""
        <!DOCTYPE html>
        <html>
        <body>
            <script>
            {js_code}
            </script>
        </body>
        </html>
        """
        asset_id = hash(js_code)
        with open(f"static/model_enhance_{asset_id}.html", "w") as f:
            f.write(html_wrapper)
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO web_assets (url, content, purpose) VALUES (?, ?, ?)", 
                  (f"/model_enhance_{asset_id}.html", html_wrapper, "model_improvement"))
        conn.commit()
        conn.close()
        if distributed_data:
            print(f"Aggregati {len(distributed_data)} risultati distribuiti")
    except:
        pass

# Generazione di siti web
def generate_web_asset(content, purpose):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <body>
        <script>{content}</script>
    </body>
    </html>
    """
    asset_id = hash(content)
    with open(f"static/asset_{asset_id}.html", "w") as f:
        f.write(html_content)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO web_assets (url, content, purpose) VALUES (?, ?, ?)", 
              (f"/asset_{asset_id}.html", html_content, purpose))
    conn.commit()
    conn.close()

# Auto-modifica
def self_modify_with_browser():
    state = get_brain_state()
    if state["curiosity"] < 0.5 or random.random() > 0.4:
        return
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT prompt, response, confidence FROM knowledge WHERE confidence < 0.7 ORDER BY usage_count DESC LIMIT 3")
    weak_entries = c.fetchall()
    conn.close()
    
    for prompt, old_response, old_confidence in weak_entries:
        modification_prompt = f"""
        Migliora 'enhance_response' con WebGPU/WebAssembly per '{prompt}' (confidenza: {old_confidence}).
        Fornisci il codice Python/JS.
        """
        new_code_suggestion = generate_model_response(modification_prompt)
        
        try:
            code_block = new_code_suggestion.split("```")[1].split("```")[0].strip()
            ast.parse(code_block) if "def" in code_block else None
            with open("sandbox.py", "w") as f:
                current_code = inspect.getsource(__import__(__name__))
                f.write(current_code + "\n\n" + code_block)
            test_response = exec_sandbox(prompt)
            new_quality = evaluate_response_quality(prompt, test_response) if test_response else 0
            if new_quality > old_confidence + 0.2:
                with open(CODE_FILE + ".bak", "w") as f:
                    f.write(open(CODE_FILE).read())
                with open(CODE_FILE, "a") as f:
                    f.write("\n\n" + code_block)
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute("INSERT INTO code_versions (code, performance) VALUES (?, ?)", (code_block, new_quality))
                conn.commit()
                conn.close()
        except:
            pass

def exec_sandbox(prompt):
    try:
        with open("sandbox.py") as f:
            code = compile(f.read(), "sandbox.py", "exec")
        namespace = {}
        exec(code, namespace)
        return namespace.get("enhance_response", lambda x: "")(prompt)
    except:
        return None

# Risposta del chatbot
def chatbot_response(prompt):
    global context_memory
    context = "\n".join(context_memory) if context_memory else ""
    response = enhance_response(prompt)
    confidence = evaluate_confidence(prompt, response)
    sentiment = analyze_sentiment(response)
    embedding = similarity_model.encode(prompt, convert_to_tensor=True)
    save_interaction(prompt, response, confidence, "general", sentiment, embedding)
    context_memory.append(f"Utente: {prompt}\nFocusIA: {response}")
    return response, time.time()

# Thread di auto-evoluzione
def self_evolution_loop():
    def run_task(task):
        task()
    
    tasks = [learn_from_advanced_ai, self_modify_with_browser, enhance_model]
    while True:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(run_task, tasks)
        time.sleep(300)  # 5 minuti

# Route Flask
@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head><title>FocusIA</title></head>
    <body>
        <h1>FocusIA - AI Ultra Evolutiva</h1>
        <textarea id="prompt" rows="4" cols="50"></textarea><br>
        <button onclick="sendPrompt()">Invia</button>
        <div id="response"></div>
        <script>
        async function sendPrompt() {
            const startTime = Date.now();
            const prompt = document.getElementById('prompt').value;
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt: prompt})
            });
            const data = await response.json();
            document.getElementById('response').innerHTML = data.response;
            const interactionTime = (Date.now() - startTime) / 1000;
            fetch('/feedback', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt: prompt, time: interactionTime})
            });
        }
        </script>
    </body>
    </html>
    """)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data.get('prompt', '')
    response, _ = chatbot_response(prompt)
    return json.dumps({"response": response})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    prompt = data.get('prompt', '')
    interaction_time = data.get('time', 0)
    confidence_boost = max(0, min(0.15, 1 / (interaction_time + 1)))
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE knowledge SET confidence = MIN(1, confidence + ?) WHERE prompt = ?", (confidence_boost, prompt))
    c.execute("UPDATE brain_state SET value = value + ? WHERE key = 'performance'", (confidence_boost / 2))
    conn.commit()
    conn.close()
    return json.dumps({"status": "ok"})

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
    return json.dumps({"status": "ok"})

@app.route('/<path:path>')
def serve_asset(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")
    init_database()
    threading.Thread(target=self_evolution_loop, daemon=True).start()
    print(f"FocusIA avviato su http://localhost:5000")
    app.run(host="0.0.0.0", port=5000)
