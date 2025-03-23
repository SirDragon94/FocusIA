import sqlite3

# Connessione al database (verrà creato se non esiste)
conn = sqlite3.connect("focusia_brain.db")
cursor = conn.cursor()

# Creazione della tabella 'knowledge'
cursor.execute("""
    CREATE TABLE IF NOT EXISTS knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        embedding BLOB,
        source TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")

# Creazione della tabella 'brain_state'
cursor.execute("""
    CREATE TABLE IF NOT EXISTS brain_state (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        state_data TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")

# Creazione della tabella 'code_versions'
cursor.execute("""
    CREATE TABLE IF NOT EXISTS code_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        code TEXT NOT NULL,
        version INTEGER NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")

# Commit delle modifiche e chiusura della connessione
conn.commit()
conn.close()

print("Database 'focusia_brain.db' creato e inizializzato con successo.")
