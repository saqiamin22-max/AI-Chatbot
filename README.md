# AI Assistant (Multi-Persona Chatbot)

A high-performance, responsive AI Chatbot interface built with Streamlit and powered by Mistral AI (`mistral-large-2512`) via LangChain. This application features a modular backend-frontend architecture and a custom, sleek dark-themed UI. It allows users to switch between specialized AI personas (Tutor, Code Expert, Data Analyst, Research Assistant, Writing Coach) or define custom behaviors on the fly.

## 🚀 Features
- **Multi-Persona Architecture:** Instantly switch between 5 pre-configured expert system prompt presets or design a custom AI behavior.
- **Real-Time Streaming:** Responses are streamed chunk-by-chunk for low latency and a smooth user experience.
- **Custom Dark UI:** Tailored with professional cyber-green aesthetics, responsive design, JetBrains Mono font integrations, and clean status badges.
- **Session Intelligence:** Tracks conversation history natively using Streamlit Session State and logs response latency (performance metrics) for every answer.
- **Data Portability:** Export your conversations instantly into either structured `JSON` files or clean, readable `TXT` logs.

## 🧱 Architecture Split
- `backend.py`: Contains constants configuration, MistralLLM streaming clients initialization, prompt orchestration, and session export logic.
- `app.py` (Frontend): Streamlit application layer handling page layout, sidebar operations, advanced custom CSS injections, and user chat state transitions.

## 🛠️ Tech Stack
- **Language:** Python
- **LLM Orchestration:** LangChain Core, LangChain MistralAI
- **Interface & UX:** Streamlit, Custom HTML/CSS
- **Environment Management:** Python-Dotenv

## 📦 How to Run

### 1. Clone the Repository
```bash
git clone [https://github.com/saqiamin22-max/YOUR_REPO_NAME.git](https://github.com/saqiamin22-max/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
