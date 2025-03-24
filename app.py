"""
FocusIA - ULTIMATE Self-Evolving AI
Copyright (C) 2025 Xhulio Guranjaku
"""

# Forzatura deploy per aggiornare il modello - 24 Mar 2025
import logging
import sys
import torch
import psutil
import sqlite3
from transformers import pipeline
import random
from collections import deque
import os
import threading
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import wikipedia
import time
import json
import concurrent.futures

# Configuriamo il logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log della memoria dopo gli import
process = psutil.Process()
mem_info_after_imports = process.memory_info()
logger.info(f"Memoria usata dopo gli import: {mem_info_after_imports.rss / 1024 / 1024:.2f} MB")

# Carica il modello tiny-gpt2 all'avvio per ottimizzare la memoria
mem_info_before = process.memory_info()
logger.info(f"Memoria usata prima del caricamento del modello: {mem_info_before.rss / 1024 / 1024:.2f} MB")

logger.info("Caricamento del modello tiny-gpt2...")
generator = pipeline('text-generation', model='sshleifer/tiny-gpt2')
logger.info("Modello caricato con successo")

mem_info_after = process.memory_info()
logger.info(f"Memoria usata dopo il caricamento del modello: {mem_info_after.rss / 1024 / 1024:.2f} MB")

from flask import Flask, request, render_template_string, send_from_directory

app = Flask(__name__)

# Inizializza il database in memoria
def init_db():
    conn = sqlite3.connect(':memory:')
    c = conn.cursor()
    c.execute("CREATE TABLE brain_state (key TEXT PRIMARY KEY, value TEXT)")
    c.execute("CREATE TABLE distributed_results (task_id TEXT, result TEXT)")
    conn.commit()
    return conn

db_conn = init_db()

# Configurazione base
model_name = "sshleifer/tiny-gpt2"
token = os.getenv("HUGGINGFACE_TOKEN")  # Token Hugging Face
context_memory = deque(maxlen=500)  # Memoria espansa
wikipedia.set_lang("it")
CODE_FILE = "focusia_ultimate_evolving.py"

# Gestore di errori globale
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Errore non gestito: {str(e)}", exc_info=True)
    return "Internal Server Error", 500

# Inizializzazione database
def init_database():
    global db_conn
    c = db_conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS knowledge 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, prompt TEXT, response TEXT, confidence REAL, 
                  usage_count INTEGER DEFAULT 0, category TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
                  sentiment REAL, cluster INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS brain_state 
                 (key TEXT PRIMARY KEY, value REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS code_versions 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT, performance REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS web_assets 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT, content TEXT, purpose TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS distributed_results 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, result TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute("INSERT OR IGNORE INTO brain_state (key, value) VALUES ('age', 0)")
    c.execute("INSERT OR IGNORE INTO brain_state (key, value) VALUES ('curiosity', 0.95)")
    c.execute("INSERT OR IGNORE INTO brain_state (key, value) VALUES ('performance', 0.5)")
    initialize_manus_knowledge(c)
    db_conn.commit()

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
        sentiment = analyze_sentiment(item["content"])
        cursor.execute("INSERT OR IGNORE INTO knowledge (prompt, response, confidence, category, sentiment) VALUES (?, ?, ?, ?, ?)", 
                       (item["topic"], item["content"], item["confidence"], "manus_base", sentiment))

# Stato mentale con metacognizione
def get_brain_state():
    c = db_conn.cursor()
    c.execute("SELECT key, value FROM brain_state")
    state = dict(c.fetchall())
    return state

def set_brain_state(key, value):
    c = db_conn.cursor()
    c.execute("INSERT OR REPLACE INTO brain_state (key, value) VALUES (?, ?)", (key, value))
    db_conn.commit()

def analyze_weaknesses():
    c = db_conn.cursor()
    c.execute("SELECT prompt, confidence FROM knowledge WHERE confidence < 0.7 ORDER BY usage_count DESC LIMIT 5")
    weak_entries = c.fetchall()
    if not weak_entries:
        return "Ottimizza la velocità di risposta e la scalabilità."
    return f"Migliora la confidenza per: {', '.join([e[0] for e in weak_entries])}"

# Salva interazioni
def save_interaction(prompt, response, confidence, category, sentiment):
    c = db_conn.cursor()
    c.execute("INSERT INTO knowledge (prompt, response, confidence, category, sentiment) VALUES (?, ?, ?, ?, ?)", 
              (prompt, response, confidence, category, sentiment))
    db_conn.commit()

# Generazione risposta
def generate_model_response(prompt):
    with torch.no_grad():  # Disabilita i gradienti per ridurre la memoria
        response = generator(prompt, max_length=50, num_return_sequences=1)
        return response[0]['generated_text']

def enhance_response(prompt):
    base_response = generate_model_response(prompt)
    c = db_conn.cursor()
    c.execute("SELECT response, confidence FROM knowledge WHERE prompt LIKE ? AND confidence > 0.9 LIMIT 1", (f"%{prompt}%",))
    best_match = c.fetchone()
    if best_match and random.random() < 0.3:
        return best_match[0] + " (Aggiornato online)"
    return base_response

# Valutazione qualità
def evaluate_confidence(prompt, response):
    prompt_words = set(prompt.lower().split())
    response_words = set(response.lower().split())
    common_words = prompt_words.intersection(response_words)
    return len(common_words) / max(len(prompt_words), 1)

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
                save_interaction(prompt, response, 0.95, "advanced_ai", 0.5)

# Distillazione avanzata
def distill_from_response(prompt, response):
    distillation_prompt = f"""
    Basandoti su: '{response[:500]}', genera una regola Python, script WebGPU o un mini-modello per migliorare FocusIA.
    Valida il codice e ottimizzalo per prestazioni.
    """
    distillation_code = generate_model_response(distillation_prompt)
    # Disabilitiamo per ora
    pass

# Ottimizzazione del modello con calcolo distribuito
def enhance_model():
    state = get_brain_state()
    if state["curiosity"] < 0.6 or random.random() > 0.4:
        return
    
    c = db_conn.cursor()
    c.execute("SELECT prompt, response FROM knowledge WHERE confidence > 0.8 LIMIT 100")
    high_conf_data = c.fetchall()
    c.execute("SELECT task_id, result FROM distributed_results WHERE timestamp > datetime('now', '-1 hour')")
    distributed_data = c.fetchall()
    
    if len(high_conf_data) < 10:
        return
    
    task_id = str(time.time())
    prompt = f"""
    Crea uno script WebGPU per ottimizzare {model_name} con questi dati: {high_conf_data[:5]}.
    Invia il risultato a '/submit_task' con task_id='{task_id}'.
    Fornisci il codice JS.
    """
    enhancement_code = generate_model_response(prompt)
    # Disabilitiamo per ora
    pass

# Generazione di siti web
def generate_web_asset(content, purpose):
    pass  # Disabilitiamo per ora

# Auto-modifica
def self_modify_with_browser():
    pass  # Disabilitiamo per ora

def exec_sandbox(prompt):
    return None  # Disabilitiamo per ora

# Risposta del chatbot
def chatbot_response(prompt):
    global context_memory
    context = "\n".join(context_memory) if context_memory else ""
    response = enhance_response(prompt)
    confidence = evaluate_confidence(prompt, response)
    sentiment = analyze_sentiment(response)
    save_interaction(prompt, response, confidence, "general", sentiment)
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
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FocusIA - AI ULTRA Evolutiva</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            margin-top: 50px;
        }
        h1 {
            font-size: 2em;
        }
        form {
            margin-top: 20px;
        }
        input[type="text"] {
            width: 300px;
            padding: 10px;
            font-size: 1em;
        }
        input[type="submit"] {
            padding: 10px 20px;
            font-size: 1em;
            margin-left: 10px;
        }
        .response {
            margin-top: 20px;
            font-size: 1.2em;
        }
    </style>
</head>
<body>
    <h1>FocusIA</h1>
    <form method="POST" action="/chat">
        <input type="text" name="prompt" placeholder="Inserisci la tua domanda">
        <input type="submit" value="Invia">
    </form>
    <div class="response">
        {% if response %}
            {{ response }}
        {% endif %}
    </div>
</body>
</html>
""")

@app.route('/chat', methods=['POST'])
def chat():
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(f"Memoria usata: {mem_info.rss / 1024 / 1024:.2f} MB")
    
    logger.info("Ricevuta richiesta POST a /chat")
    prompt = request.form.get('prompt', '')
    logger.info(f"Prompt ricevuto: {prompt}")
    try:
        logger.info("Inizio generazione risposta")
        response = generate_model_response(prompt)
        logger.info(f"Risposta generata: {response}")
        logger.info("Inizio chiamata a chatbot_response")
        response, _ = chatbot_response(prompt)
        logger.info("chatbot_response completato")
    except Exception as e:
        logger.error(f"Errore durante la generazione della risposta: {str(e)}", exc_info=True)
        raise
    logger.info("Inizio rendering del template")
    return render_template_string("""
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FocusIA - AI ULTRA Evolutiva</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            margin-top: 50px;
        }
        h1 {
            font-size: 2em;
        }
        form {
            margin-top: 20px;
        }
        input[type="text"] {
            width: 300px;
            padding: 10px;
            font-size: 1em;
        }
        input[type="submit"] {
            padding: 10px 20px;
            font-size: 1em;
            margin-left: 10px;
        }
        .response {
            margin-top: 20px;
            font-size: 1.2em;
        }
    </style>
</head>
<body>
    <h1>FocusIA</h1>
    <form method="POST" action="/chat">
        <input type="text" name="route prompt" placeholder="Inserisci la tua domanda">
        <input type="submit" value="Invia">
    </form>
    <div class="response">
        {{ response }}
    </div [response]
    </div>
</body>
</html>
""", response=response)

@app.route('/submit_task', methods=['POST'])
def submit_task():
    data = request.get_json()
    task_id = data.get('task_id', '')
    result = data.get('result', '')
    c = db_conn.cursor()
    c.execute("INSERT INTO distributed_results (task_id, result) VALUES (?, ?)", (task_id, result))
    db_conn.commit()
    return json.dumps({"status": "ok"})

@app.route('/<path:path>')
def serve_asset(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")
    init_database()
    threading.Thread(target=self_evolution_loop, daemon=True).start()
    logger.info("FocusIA avviato su http://localhost:5000")
    app.run(host="0.0.0.0", port=5000)
