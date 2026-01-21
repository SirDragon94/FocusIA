# FocusIA - Render + Supabase

Assistente AI evolutivo leggero deployato su Render free tier con backend Supabase.

- Frontend: Dashboard con chat, timer Pomodoro, task, upload PDF
- Backend: Flask + Supabase (database Postgres)
- AI: API OpenAI + Hugging Face fallback
- Evoluzione: Prompt evolution, feedback loop, curiosity learning

Env vars necessarie:
- OPENAI_API_KEY
- HUGGINGFACE_TOKEN
- SUPABASE_URL
- SUPABASE_KEY

## Requisiti

- Python 3.9+
- Librerie elencate in `requirements.txt`:
  - Flask
  - Transformers
  - Sentence-Transformers
  - Torch
  - Requests
  - BeautifulSoup4
  - Googlesearch-python
  - Wikipedia-api
  - Numpy
  - Schedule
  - pypdf

## Installazione locale

1. Clona il repository:
   ```bash
   git clone https://github.com/SirDragon94/FocusIA.git
   cd FocusIA
   ```
2. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```
3. Avvia l'applicazione:
   ```bash
   python app.py
   ```

## Copyright

Copyright (C) 2025 Xhulio Guranjaku. Tutti i diritti riservati.
