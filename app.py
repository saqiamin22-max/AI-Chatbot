import streamlit as st
import time
import datetime
from langchain_core.messages import HumanMessage, AIMessage

# Backend functions ko import karna
from backend import (
    SYSTEM_PRESETS,
    get_mistral_model,
    prepare_messages,
    export_to_json,
    export_to_txt
)

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme Styling
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
    border-radius: 6px !important;
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
}
code, pre {
    background: #060A06 !important;
    border: 1px solid #1A2E1A !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: #86EFAC !important;
}
.status-badge {
    display: inline-block;
    background: rgba(74,222,128,0.1);
    border: 1px solid rgba(74,222,128,0.3);
    color: #4ADE80;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 10px;
    border-radius: 20px;
}
</style>
""", unsafe_allow_html=True)


# Session State Initialization
def init_state():
    defaults = {
        "messages": [],
        "system_prompt": SYSTEM_PRESETS["🎓 AI Tutor"],
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
    st.markdown("## 🔥 AI Assistant")
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
            st.session_state.response_times = []

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
        st.rerun()

    if st.button("📥 Export Chat (JSON)", use_container_width=True):
        json_data = export_to_json(st.session_state.messages)
        st.download_button(
            label="💾 Download JSON",
            data=json_data,
            file_name=f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    if st.button("📄 Export Chat (TXT)", use_container_width=True):
        txt_data = export_to_txt(st.session_state.messages, st.session_state.session_start)
        st.download_button(
            label="💾 Download TXT",
            data=txt_data,
            file_name=f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

# ─────────────────────────────────────────────
# MAIN CHAT INTERFACE
# ─────────────────────────────────────────────
st.markdown("# 🔥 AI Assistant")

persona_icon = st.session_state.preset.split()[0]
st.markdown(
    f'<div style="background:#0D140D;border:1px solid #2D4A2D;border-radius:8px;'
    f'padding:0.5rem 1rem;margin-bottom:1rem;font-family:JetBrains Mono,monospace;'
    f'font-size:0.78rem;color:#4ADE80;letter-spacing:0.05em;">'
    f'{persona_icon} ACTIVE PERSONA: <span style="color:#F59E0B;">{st.session_state.preset}</span>'
    f'</div>',
    unsafe_allow_html=True
)

# Chat History Rendering
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
                rt_index = sum(1 for m in st.session_state.messages[:i] if isinstance(m, AIMessage))
                if rt_index < len(st.session_state.response_times):
                    rt = st.session_state.response_times[rt_index]
                    st.markdown(
                        f'<div style="font-family:JetBrains Mono;font-size:0.65rem;color:#4A5568;">⏱ {rt}s</div>',
                        unsafe_allow_html=True)

# User Chat Input Handling
user_input = st.chat_input("Type message here...")

if user_input and user_input.strip():
    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.message_count += 1

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        start_time = time.time()

        try:
            model = get_mistral_model()
            formatted_msgs = prepare_messages(st.session_state.system_prompt, st.session_state.messages)

            for chunk in model.stream(formatted_msgs):
                if chunk.content:
                    full_response += chunk.content
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)
            elapsed = round(time.time() - start_time, 2)
            st.session_state.response_times.append(elapsed)

            st.markdown(f'<div style="font-family:JetBrains Mono;font-size:0.65rem;color:#4A5568;">⏱ {elapsed}s</div>',
                        unsafe_allow_html=True)

        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}"
            placeholder.error(error_msg)
            full_response = error_msg

    st.session_state.messages.append(AIMessage(content=full_response))
    st.session_state.message_count += 1