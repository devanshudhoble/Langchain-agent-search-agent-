import os
import sys
from pathlib import Path
import yaml
import streamlit as st

# Make sure src is in python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_env
# Load environment variables (.env)
load_env()

from src.agent import SearchAgent
from src.docs_loader import load_documents
from src.vector_store import get_faiss_index

# Load configuration
config_path = Path(__file__).parent.parent / "config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# Set Streamlit Page Configuration with B&W / dark theme preference
st.set_page_config(
    page_title="Gemini Search Agent",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Dark Black-and-White Custom styling
st.markdown(
    """
    <style>
    /* Professional Black-and-White / Minimalist Theme overrides */
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
    /* Custom separator line */
    hr {
        border-color: #222222 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("LangChain Gemini Search Agent")
st.caption("Powered by Google Gemini and FAISS Vector Store")
st.markdown("---")

# Make sure data directory exists
data_dir = Path(cfg["loader"]["source_path"])
if not data_dir.exists():
    data_dir.mkdir(parents=True, exist_ok=True)

# Load documents and create/fetch vector store
if "vector_store" not in st.session_state:
    docs = load_documents(str(data_dir), cfg["loader"]["allowed_extensions"])
    if not docs:
        st.warning("No documents found in the `data/` directory. Please place PDF, TXT, or MD files there.")
        st.stop()
    vector_store = get_faiss_index(docs, index_path=cfg["vector_store"]["index_path"])
    st.session_state.vector_store = vector_store
else:
    vector_store = st.session_state.vector_store

# Initialize the agent
if "agent" not in st.session_state:
    agent = SearchAgent(vector_store=vector_store, llm_cfg=cfg["llm"])
    st.session_state.agent = agent
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

user_input = st.chat_input("Ask a question about your documents")
if user_input:
    # Display user's input immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents & generating response..."):
            answer = agent.run(user_input)
            st.markdown(answer)
            
    st.session_state.chat_history.append((user_input, answer))
