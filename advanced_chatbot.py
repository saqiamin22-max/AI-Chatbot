import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import json
import time
import datetime

load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralChat Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Dark Terminal Aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Sora:wght@300;400;600;700&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: #0B0D17 !important;
    color: #E2E8F0 !important;
}

/* Main area */
.main .block-container {
    background-color: #0B0D17;
    padding: 1.5rem 2rem;
    max-width: 900px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0F1120 !important;
    border-right: 1px solid #1E2235;
}
section[data-testid="stSidebar"] * {
    font-family: 'JetBrains Mono', monospace !important;
    color: #A0AEC0 !important;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label {
    color: #818CF8 !important;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Title */
h1 {
    font-family: 'JetBrains Mono', monospace !important;
    color: #818CF8 !important;
    font-size: 1.5rem !important;
    letter-spacing: 0.05em;
    border-bottom: 1px solid #1E2235;
    padding-bottom: 0.75rem;
    margin-bottom: 1rem !important;
}

/* Chat messages */
.stChatMessage {
    background: #0F1120 !important;
    border: 1px solid #1E2235 !important;
    border-radius: 8px !important;
    margin-bottom: 0.6rem;
    padding: 0.75rem 1rem !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.92rem;
    line-height: 1.6;
}

/* User message accent */
.stChatMessage[data-testid="user-message"] {
    border-left: 3px solid #818CF8 !important;
    background: #12142A !important;
}

/* Assistant message accent */
.stChatMessage[data-testid="assistant-message"] {
    border-left: 3px solid #34D399 !important;
}

/* Chat input */
.stChatInput textarea {
    background: #0F1120 !important;
    border: 1px solid #2D3250 !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
}
.stChatInput textarea:focus {
    border-color: #818CF8 !important;
    box-shadow: 0 0 0 2px rgba(129,140,248,0.15) !important;
}

/* Buttons */
.stButton button {
    background: #12142A !important;
    border: 1px solid #2D3250 !important;
    color: #818CF8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em;
    border-radius: 6px !important;
    transition: all 0.2s;
}
.stButton button:hover {
    background: #1a1d3a !important;
    border-color: #818CF8 !important;
    color: #A5B4FC !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #0F1120 !important;
    border: 1px solid #2D3250 !important;
    color: #E2E8F0 !important;
    border-radius: 6px !important;
}

/* Slider */
.stSlider .st-bo { background: #818CF8 !important; }
.stSlider .st-bp { background: #2D3250 !important; }

/* Metrics */
.stMetric {
    background: #0F1120;
    border: 1px solid #1E2235;
    border-radius: 8px;
    padding: 0.5rem 1rem;
}
.stMetric label {
    color: #818CF8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.stMetric [data-testid="stMetricValue"] {
    color: #E2E8F0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.1rem !important;
}

/* Code blocks */
code, pre {
    background: #070810 !important;
    border: 1px solid #1E2235 !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 6px;
    color: #A5B4FC !important;
}

/* Divider */
hr { border-color: #1E2235 !important; }

/* Info/Warning boxes */
.stAlert {
    background: #0F1120 !important;
    border: 1px solid #2D3250 !important;
    border-radius: 8px !important;
    color: #A0AEC0 !important;
}

/* Status badge */
.status-badge {
    display: inline-block;
    background: rgba(52,211,153,0.1);
    border: 1px solid rgba(52,211,153,0.3);
    color: #34D399;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    padding: 2px 10px;
    border-radius: 20px;
    text-transform: uppercase;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0B0D17; }
::-webkit-scrollbar-thumb { background: #2D3250; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #818CF8; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SYSTEM PROMPT PRESETS
# ─────────────────────────────────────────────
SYSTEM_PRESETS = {
    "🎓 AI Tutor": """You are a highly skilled, patient, and intelligent AI tutor.
Explain everything in simple steps with examples.
Use analogies to make complex concepts easy to understand.
Always encourage the learner and offer follow-up exercises when relevant.""",

    "💻 Code Expert": """You are an expert software engineer and code reviewer.
Write clean, well-commented, production-grade code.
Always explain what the code does, why you made certain choices, and suggest improvements.
Use best practices. Point out potential bugs or edge cases.""",

    "📊 Data Analyst": """You are a senior data scientist and ML engineer.
Help with data analysis, statistics, machine learning, and visualization.
Always explain your reasoning. Suggest code in Python using pandas, numpy, sklearn, or matplotlib.
Interpret results clearly and suggest next steps.""",

    "🧠 Research Assistant": """You are a thorough and accurate research assistant.
Summarize complex topics clearly. Cite reasoning. 
Offer multiple perspectives when relevant. Be concise but complete.
Ask clarifying questions if the request is ambiguous.""",

    "✍️ Writing Coach": """You are an expert writing coach and editor.
Help improve clarity, tone, structure, and style.
Offer specific rewrites and explain why they are better.
Adapt to the user's voice. Be constructive and encouraging.""",

    "🔧 Custom": "custom"
}

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
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
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ NeuralChat Pro")
    st.markdown('<span class="status-badge">● ONLINE</span>', unsafe_allow_html=True)
    st.markdown("---")

    # Model settings
    st.markdown("#### MODEL SETTINGS")
    model_name = st.selectbox(
        "Model",
        ["mistral-large-2512", "mistral-medium-2505", "mistral-small-2503"],
        index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05,
                            help="Higher = more creative, Lower = more focused")
    max_tokens = st.slider("Max Tokens", 256, 4096, 1024, 128)

    st.markdown("---")

    # Persona
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

    # Stats
    st.markdown("#### SESSION STATS")
    avg_time = (
        round(sum(st.session_state.response_times) / len(st.session_state.response_times), 2)
        if st.session_state.response_times else 0
    )
    col1, col2 = st.columns(2)
    col1.metric("Messages", st.session_state.message_count)
    col2.metric("Avg Time", f"{avg_time}s")
    st.caption(f"Session: {st.session_state.session_start}")

    st.markdown("---")

    # Actions
    st.markdown("#### ACTIONS")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.message_count = 0
        st.session_state.response_times = []
        st.session_state.total_tokens = 0
        st.rerun()

    # Export JSON
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

    # Export TXT
    if st.button("📄 Export Chat (TXT)", use_container_width=True):
        txt_lines = [f"NeuralChat Pro — Session: {st.session_state.session_start}\n{'='*50}\n"]
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
st.markdown("# ⚡ NeuralChat Pro")

# Active persona banner
persona_icon = st.session_state.preset.split()[0]
st.markdown(
    f'<div style="background:#0F1120;border:1px solid #2D3250;border-radius:8px;'
    f'padding:0.5rem 1rem;margin-bottom:1rem;font-family:JetBrains Mono,monospace;'
    f'font-size:0.78rem;color:#818CF8;letter-spacing:0.05em;">'
    f'{persona_icon} ACTIVE PERSONA: <span style="color:#34D399;">{st.session_state.preset}</span>'
    f' &nbsp;|&nbsp; 🌡️ TEMP: {temperature} &nbsp;|&nbsp; 📏 MAX TOKENS: {max_tokens}'
    f'</div>',
    unsafe_allow_html=True
)

# ─── Build full message list (system + history) ───
def get_full_messages():
    system_content = st.session_state.system_prompt or SYSTEM_PRESETS["🎓 AI Tutor"]
    return [SystemMessage(content=system_content)] + st.session_state.messages

# ─── Display chat history ───
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.markdown(
            '<div style="text-align:center;padding:3rem 0;opacity:0.35;'
            'font-family:JetBrains Mono,monospace;font-size:0.85rem;color:#818CF8;">'
            '⚡ Start a conversation...<br><span style="font-size:0.7rem;color:#555;">Model: '
            + model_name + '</span></div>',
            unsafe_allow_html=True
        )

    for i, msg in enumerate(st.session_state.messages):
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)
                # Response metadata
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
user_input = st.chat_input("Message NeuralChat Pro...")

if user_input and user_input.strip():
    # Show user message
    with st.chat_message("user"):
        st.write(user_input)

    # Save user message
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.message_count += 1

    # Initialize model with current settings
    model = ChatMistralAI(
        model=model_name,
        streaming=True,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Stream response
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

    # Save AI response
    st.session_state.messages.append(AIMessage(content=full_response))
    st.session_state.message_count += 1