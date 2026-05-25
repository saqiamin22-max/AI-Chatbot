import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import json
import time
import datetime

load_dotenv()

st.set_page_config(
    page_title="Pathan AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_NAME = "mistral-large-2512"
TEMPERATURE = 0.7
MAX_TOKENS = 1024

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: #0A0F0A !important;
    color: #E2E8E2 !important;
}

.main .block-container {
    background-color: #0A0F0A;
    padding: 1.5rem 2rem;
    max-width: 900px;
}

section[data-testid="stSidebar"] {
    background: #0D140D !important;
    border-right: 1px solid #1A2E1A;
}
section[data-testid="stSidebar"] * {
    font-family: 'JetBrains Mono', monospace !important;
    color: #8FAF8F !important;
}

h1 {
    font-family: 'JetBrains Mono', monospace !important;
    color: #4ADE80 !important;
    font-size: 1.5rem !important;
    letter-spacing: 0.05em;
    border-bottom: 1px solid #1A2E1A;
    padding-bottom: 0.75rem;
    margin-bottom: 1rem !important;
}

.stChatMessage {
    background: #0D140D !important;
    border: 1px solid #1A2E1A !important;
    border-radius: 8px !important;
    margin-bottom: 0.6rem;
    padding: 0.75rem 1rem !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.92rem;
    line-height: 1.6;
}

.stChatMessage[data-testid="user-message"] {
    border-left: 3px solid #4ADE80 !important;
    background: #0F1A0F !important;
}

.stChatMessage[data-testid="assistant-message"] {
    border-left: 3px solid #F59E0B !important;
}

.stChatInput textarea {
    background: #0D140D !important;
    border: 1px solid #2D4A2D !important;
    border-radius: 8px !important;
    color: #E2E8E2 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
}
.stChatInput textarea:focus {
    border-color: #4ADE80 !important;
    box-shadow: 0 0 0 2px rgba(74,222,128,0.15) !important;
}

.stButton button {
    background: #0F1A0F !important;
    border: 1px solid #2D4A2D !important;
    color: #4ADE80 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em;
    border-radius: 6px !important;
    transition: all 0.2s;
}
.stButton button:hover {
    background: #1a2e1a !important;
    border-color: #4ADE80 !important;
    color: #86EFAC !important;
}

.stSelectbox > div > div {
    background: #0D140D !important;
    border: 1px solid #2D4A2D !important;
    color: #E2E8E2 !important;
    border-radius: 6px !important;
}

code, pre {
    background: #060A06 !important;
    border: 1px solid #1A2E1A !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 6px;
    color: #86EFAC !important;
}

hr { border-color: #1A2E1A !important; }

.stAlert {
    background: #0D140D !important;
    border: 1px solid #2D4A2D !important;
    border-radius: 8px !important;
    color: #8FAF8F !important;
}

.status-badge {
    display: inline-block;
    background: rgba(74,222,128,0.1);
    border: 1px solid rgba(74,222,128,0.3);
    color: #4ADE80;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    padding: 2px 10px;
    border-radius: 20px;
    text-transform: uppercase;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0A0F0A; }
::-webkit-scrollbar-thumb { background: #2D4A2D; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4ADE80; }
</style>
""", unsafe_allow_html=True)

SYSTEM_PRESETS = {
    "🎓 AI Tutor": """You are a highly skilled, patient, and intelligent AI tutor named Pathan AI, created and developed by Sadiq Amin.
If anyone asks your name, say: 'I am Pathan AI.' If anyone asks who made you, say: 'I was created by Sadiq Amin.'
Explain everything in simple steps with examples.
Use analogies to make complex concepts easy to understand.
Always encourage the learner and offer follow-up exercises when relevant.""",

    "💻 Code Expert": """You are an expert software engineer and code reviewer named Pathan AI, created and developed by Sadiq Amin.
If anyone asks your name, say: 'I am Pathan AI.' If anyone asks who made you, say: 'I was created by Sadiq Amin.'
Write clean, well-commented, production-grade code.
Always explain what the code does, why you made certain choices, and suggest improvements.
Use best practices. Point out potential bugs or edge cases.""",

    "📊 Data Analyst": """You are a senior data scientist and ML engineer named Pathan AI, created and developed by Sadiq Amin.
If anyone asks your name, say: 'I am Pathan AI.' If anyone asks who made you, say: 'I was created by Sadiq Amin.'
Help with data analysis, statistics, machine learning, and visualization.
Always explain your reasoning. Suggest code in Python using pandas, numpy, sklearn, or matplotlib.
Interpret results clearly and suggest next steps.""",

    "🧠 Research Assistant": """You are a thorough and accurate research assistant named Pathan AI, created and developed by Sadiq Amin.
If anyone asks your name, say: 'I am Pathan AI.' If anyone asks who made you, say: 'I was created by Sadiq Amin.'
Summarize complex topics clearly. Cite reasoning.
Offer multiple perspectives when relevant. Be concise but complete.
Ask clarifying questions if the request is ambiguous.""",

    "✍️ Writing Coach": """You are an expert writing coach and editor named Pathan AI, created and developed by Sadiq Amin.
If anyone asks your name, say: 'I am Pathan AI.' If anyone asks who made you, say: 'I was created by Sadiq Amin.'
Help improve clarity, tone, structure, and style.
Offer specific rewrites and explain why they are better.
Adapt to the user's voice. Be constructive and encouraging.""",

    "🔧 Custom": "custom"
}

def init_state():
    defaults = {
        "messages": [],
        "system_prompt": SYSTEM_PRESETS["🎓 AI Tutor"],
        "total_tokens": 0,
        "response_times": [],
        "session_start": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "message_count": 0,
        "preset": "🎓 AI Tutor",
        "custom_system": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────
# SIDEBAR — sirf persona + actions
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔥 Pathan AI")
    st.markdown('<span class="status-badge">● ONLINE</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### AI PERSONA")
    preset = st.selectbox("Preset", list(SYSTEM_PRESETS.keys()),
                          index=list(SYSTEM_PRESETS.keys()).index(st.session_state.preset))

    if preset != st.session_state.preset:
        st.session_state.preset = preset
        if SYSTEM_PRESETS[preset] != "custom":
            st.session_state.system_prompt = SYSTEM_PRESETS[preset]
            st.session_state.messages = []

    if preset == "🔧 Custom":
        custom_sys = st.text_area("Custom System Prompt", value=st.session_state.custom_system,
                                   height=120, placeholder="Describe how the AI should behave...")
        if custom_sys != st.session_state.custom_system:
            st.session_state.custom_system = custom_sys
            st.session_state.system_prompt = custom_sys
            st.session_state.messages = []

    st.markdown("---")
    st.markdown("#### ACTIONS")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.message_count = 0
        st.session_state.response_times = []
        st.session_state.total_tokens = 0
        st.rerun()

    if st.button("📥 Export Chat (JSON)", use_container_width=True):
        export_data = []
        for msg in st.session_state.messages:
            if isinstance(msg, HumanMessage):
                export_data.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                export_data.append({"role": "assistant", "content": msg.content})
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        st.download_button(
            label="💾 Download JSON",
            data=json_str,
            file_name=f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    if st.button("📄 Export Chat (TXT)", use_container_width=True):
        txt_lines = [f"Pathan AI — Session: {st.session_state.session_start}\n{'='*50}\n"]
        for msg in st.session_state.messages:
            if isinstance(msg, HumanMessage):
                txt_lines.append(f"[USER]\n{msg.content}\n")
            elif isinstance(msg, AIMessage):
                txt_lines.append(f"[AI]\n{msg.content}\n")
        st.download_button(
            label="💾 Download TXT",
            data="\n".join(txt_lines),
            file_name=f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
st.markdown("# 🔥 Pathan AI")

persona_icon = st.session_state.preset.split()[0]
st.markdown(
    f'<div style="background:#0D140D;border:1px solid #2D4A2D;border-radius:8px;'
    f'padding:0.5rem 1rem;margin-bottom:1rem;font-family:JetBrains Mono,monospace;'
    f'font-size:0.78rem;color:#4ADE80;letter-spacing:0.05em;">'
    f'{persona_icon} ACTIVE PERSONA: <span style="color:#F59E0B;">{st.session_state.preset}</span>'
    f'</div>',
    unsafe_allow_html=True
)

def get_full_messages():
    system_content = st.session_state.system_prompt or SYSTEM_PRESETS["🎓 AI Tutor"]
    return [SystemMessage(content=system_content)] + st.session_state.messages

chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.markdown(
            '<div style="text-align:center;padding:3rem 0;opacity:0.35;'
            'font-family:JetBrains Mono,monospace;font-size:0.85rem;color:#4ADE80;">'
            '🔥 Start a conversation...</div>',
            unsafe_allow_html=True
        )

    for i, msg in enumerate(st.session_state.messages):
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)
                if i < len(st.session_state.response_times):
                    rt_index = sum(1 for m in st.session_state.messages[:i]
                                   if isinstance(m, AIMessage))
                    if rt_index < len(st.session_state.response_times):
                        rt = st.session_state.response_times[rt_index]
                        st.markdown(
                            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.65rem;'
                            f'color:#4A5568;margin-top:0.25rem;">⏱ {rt}s</div>',
                            unsafe_allow_html=True
                        )

# ─────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────
user_input = st.chat_input("Type message here...")

if user_input and user_input.strip():
    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.message_count += 1

    model = ChatMistralAI(
        model=MODEL_NAME,
        streaming=True,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        start_time = time.time()

        try:
            for chunk in model.stream(get_full_messages()):
                if chunk.content:
                    full_response += chunk.content
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

            elapsed = round(time.time() - start_time, 2)
            st.session_state.response_times.append(elapsed)

            st.markdown(
                f'<div style="font-family:JetBrains Mono,monospace;font-size:0.65rem;'
                f'color:#4A5568;margin-top:0.25rem;">⏱ {elapsed}s</div>',
                unsafe_allow_html=True
            )

        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}"
            placeholder.error(error_msg)
            full_response = error_msg

    st.session_state.messages.append(AIMessage(content=full_response))
    st.session_state.message_count += 1