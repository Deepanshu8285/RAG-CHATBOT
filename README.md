# 📚 Chat with Multiple PDFs — Multimodal RAG Chatbot

A full-stack Retrieval-Augmented Generation (RAG) chatbot that lets users upload PDFs (text, images, and tables) and ask natural-language questions grounded in their content — with user accounts, persistent per-user document storage, source citations, and streaming responses.

Built as a hands-on project to learn the full RAG pipeline: document parsing, embeddings, vector search, prompt engineering, and multimodal (vision) retrieval — combined with a real authentication system and a polished chat UI.

---

## ✨ Features

- **Multimodal document understanding**
  - Extracts and indexes plain text from PDFs
  - Extracts embedded images/diagrams and describes them using GPT-4o vision
  - Detects tables and transcribes them into structured text
- **Conversational RAG** with short-term memory (remembers recent turns for natural follow-up questions)
- **Source citations** — every answer shows which document/page it was grounded in
- **Streaming responses** — answers appear token-by-token, like a real chat app
- **User accounts** — sign up, log in, stay logged in via secure cookies (7-day sessions)
- **Per-user persistence** — each user's documents and vector index are saved to disk and reload automatically across sessions
- **Document management** — delete individual documents from your knowledge base without rebuilding everything
- **Download conversation** — export any chat as a `.txt` transcript
- **Rate-limit safety** — filters out low-value images, caps visuals per document, and paces API calls to avoid quota errors

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| LLM & Embeddings | OpenAI GPT-4o-mini (chat + vision), `text-embedding` via `langchain-openai` |
| Retrieval | FAISS (vector similarity search), LangChain |
| PDF Processing | `pypdf` (text), `PyMuPDF` / `fitz` (images), `pdfplumber` (table detection) |
| Auth | `streamlit-authenticator` (hashed passwords, cookie sessions) |
| Frontend | Streamlit (`st.chat_message`, `st.chat_input`, `st.write_stream`) |
| Storage | Local filesystem (FAISS index + extracted images, per user) |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Deepanshu8285/RAG-CHATBOT.git
cd RAG-CHATBOT
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your OpenAI API key
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-api-key-here
```
Get a key at [platform.openai.com](https://platform.openai.com/api-keys). You'll need billing enabled — costs for this app are minimal (typically a few cents even with heavy testing).

### 5. Set up authentication config
```bash
cp config.yaml.example config.yaml
```
Then either:
- Use the app's built-in **Sign Up** tab to create your first account (recommended), or
- Manually generate a password hash and edit `config.yaml`:
```bash
python3 -c "import streamlit_authenticator as stauth; print(stauth.Hasher().hash('yourpassword'))"
```

### 6. Run the app
```bash
streamlit run app.py
```
Open the local URL shown in your terminal (typically `http://localhost:8501`).

---

## 📖 How It Works

1. **Upload** one or more PDFs through the sidebar and click **Process**.
2. The app extracts plain text, embedded images, and tables from each document.
3. Images and tables are described/transcribed using GPT-4o vision and stored alongside the plain text.
4. All content is chunked, embedded, and indexed in a FAISS vector store — saved to disk under your account.
5. Ask a question — the app retrieves the most relevant chunks (text, image descriptions, or table data), and GPT-4o-mini generates a grounded answer, streamed back token-by-token with source citations.
6. Come back later — your documents and account are still there.

---

## ⚠️ Known Limitations

- **Local storage only**: documents and the vector index are saved to the local filesystem. On ephemeral hosting (e.g. free-tier Streamlit Community Cloud), this storage may not persist across app restarts — a production deployment would use a proper database or cloud object storage instead.
- **Image relevance depends on the PDF**: purely decorative images (backgrounds, logos) get described like any other image; a size filter reduces this but doesn't eliminate it entirely.
- **No "forgot password" flow** or username/email-interchangeable login yet — noted as a possible future improvement.

---

## 🔮 Possible Future Improvements

- Hybrid search (keyword + vector) for better retrieval accuracy
- Cloud-based persistent storage for real multi-user production deployment
- Password reset via email
- Multi-document comparison mode
- Evaluation metrics for answer quality

---

## 🙏 Acknowledgements

Built while learning RAG, LangChain, and multimodal LLM applications — with heavy iterative debugging along the way (deprecated library APIs, GPU/quantization constraints from an earlier LLM fine-tuning phase of this project, rate limits, and version mismatches), which turned out to be as valuable a learning experience as the final feature set.
