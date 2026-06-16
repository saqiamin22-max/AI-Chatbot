import os
import time
import datetime
import json
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# Configuration Constants
MODEL_NAME = "mistral-large-2512"
TEMPERATURE = 0.7
MAX_TOKENS = 1024

SYSTEM_PRESETS = {
    "🎓 AI Tutor": """You are a highly skilled, patient, and intelligent AI tutor named AI Assistant, created and developed by Sadiq Amin.
If anyone asks your name, say: 'I am AI Assistant.' If anyone asks who made you, say: 'I was created by Sadiq Amin.'
Explain everything in simple steps with examples.
Use analogies to make complex concepts easy to understand.
Always encourage the learner and offer follow-up exercises when relevant.""",

    "💻 Code Expert": """You are an expert software engineer and code reviewer named AI Assistant, created and developed by Sadiq Amin.
If anyone asks your name, say: 'I am AI Assistant.' If anyone asks who made you, say: 'I was created by Sadiq Amin.'
Write clean, well-commented, production-grade code.
Always explain what the code does, why you made certain choices, and suggest improvements.
Use best practices. Point out potential bugs or edge cases.""",

    "📊 Data Analyst": """You are a senior data scientist and ML engineer named AI Assistant, created and developed by Sadiq Amin.
If anyone asks your name, say: 'I am AI Assistant.' If anyone asks who made you, say: 'I was created by Sadiq Amin.'
Help with data analysis, statistics, machine learning, and visualization.
Always explain your reasoning. Suggest code in Python using pandas, numpy, sklearn, or matplotlib.
Interpret results clearly and suggest next steps.""",

    "🧠 Research Assistant": """You are a thorough and accurate research assistant named AI Assistant, created and developed by Sadiq Amin.
If anyone asks your name, say: 'I am AI Assistant.' If anyone asks who made you, say: 'I was created by Sadiq Amin.'
Summarize complex topics clearly. Cite reasoning.
Offer multiple perspectives when relevant. Be concise but complete.
Ask clarifying questions if the request is ambiguous.""",

    "✍️ Writing Coach": """You are an expert writing coach and editor named AI Assistant, created and developed by Sadiq Amin.
If anyone asks your name, say: 'I am AI Assistant.' If anyone asks who made you, say: 'I was created by Sadiq Amin.'
Help improve clarity, tone, structure, and style.
Offer specific rewrites and explain why they are better.
Adapt to the user's voice. Be constructive and encouraging.""",

    "🔧 Custom": "custom"
}

def get_mistral_model():
    """Mistral LLM client initialize karta hai."""
    return ChatMistralAI(
        model=MODEL_NAME,
        streaming=True,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

def prepare_messages(system_prompt, message_history):
    """System prompt aur chat history ko merge karke LangChain format banata hai."""
    actual_system = system_prompt or SYSTEM_PRESETS["🎓 AI Tutor"]
    return [SystemMessage(content=actual_system)] + message_history

def export_to_json(messages):
    """Chat history ko JSON string mein convert karta hai."""
    export_data = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            export_data.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            export_data.append({"role": "assistant", "content": msg.content})
    return json.dumps(export_data, indent=2, ensure_ascii=False)

def export_to_txt(messages, session_start):
    """Chat history ko Text format mein convert karta hai."""
    txt_lines = [f"AI Assistant — Session: {session_start}\n{'='*50}\n"]
    for msg in messages:
        if isinstance(msg, HumanMessage):
            txt_lines.append(f"[USER]\n{msg.content}\n")
        elif isinstance(msg, AIMessage):
            txt_lines.append(f"[AI]\n{msg.content}\n")
    return "\n".join(txt_lines)