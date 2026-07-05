import asyncio
import os
import sys
from pathlib import Path
import yaml
import streamlit as st

# Setup event loop for the thread
try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Make sure src is in python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_env
# Load environment variables (.env)
load_env()

from src.agent import SearchAgent

# Load configuration
config_path = Path(__file__).parent.parent / "config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="LangChain Web Search Agent",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Dark B&W Custom styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0c0c0c;
        color: #f5f5f5;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .stChatInputContainer {
        border-color: #333333 !important;
    }
    .stChatInputContainer input {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
    }
    /* Style assistant messages */
    div[data-testid="stChatMessage"] {
        background-color: #141414;
        border: 1px solid #222222;
        border-radius: 8px;
        margin-bottom: 12px;
        padding: 15px;
    }
    /* Style user messages differently */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageUserIcon"]) {
        background-color: #1e1e1e;
        border: 1px solid #333333;
    }
    hr {
        border-color: #222222 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("LangChain Web Search Agent")
st.caption("Powered by Google Gemini & DuckDuckGo Search")
st.markdown("---")

# Initialize the agent
if "agent" not in st.session_state:
    try:
        agent = SearchAgent(llm_cfg=cfg["llm"])
        st.session_state.agent = agent
    except Exception as e:
        st.error(f"Failed to initialize Agent: {e}")
        st.stop()
else:
    agent = st.session_state.agent

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)

user_input = st.chat_input("Ask me anything (I can search the web in real-time!)")
if user_input:
    # Display user's input immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Agent searching the web & reasoning..."):
            try:
                answer = agent.run(user_input)
                st.markdown(answer)
                st.session_state.chat_history.append((user_input, answer))
            except Exception as e:
                st.error(f"An error occurred: {e}")
