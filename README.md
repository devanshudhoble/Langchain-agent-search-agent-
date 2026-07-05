# LangChain Web Search Agent

## Overview
This project is an intelligent **Web Search AI Agent** built using the **LangChain** framework and **Google Gemini** (`gemini-1.5-flash`). It is equipped with a DuckDuckGo Search tool, allowing it to search the web in real-time, gather up-to-date information, and reason about it to provide accurate answers.

It features:
- **LangChain Agent Framework**: Uses a tool-calling agent configuration (`create_tool_calling_agent`).
- **Real-time Web Access**: DuckDuckGo Search integration for live factual lookup.
- **Conversational Memory**: Remembers chat context using `InMemoryChatMessageHistory`.
- **Streamlit Interface**: A gorgeous, minimal dark/black-and-white chat UI.
- **Dockerized Ready**: Included Dockerfile for containerized deployments.

## Resume Bullet Point
> **Engineered a LangChain-based AI Search Agent integrating Google Gemini with DuckDuckGo Search API to enable real-time web retrieval. Developed conversational memory pipelines using LangChain Memory modules and wrapped the system in a responsive dark-themed Streamlit UI. Containerized the codebase with Docker to streamline local testing and cloud deployment.**

## Quick Start (Local)
1. **Clone the repository**:
   ```bash
   git clone https://github.com/devanshudhoble/Langchain-agent-search-agent-.git
   cd Langchain-agent-search-agent-
   ```
2. **Setup virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure your API Key**:
   Create a `.env` file in the root folder:
   ```text
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
5. **Run the Streamlit application**:
   ```bash
   streamlit run src/main.py
   ```
   Open the printed **Local URL** (e.g. `http://localhost:8501`) in your browser.

## Docker Deployment
```bash
docker build -t langchain-search-agent .
docker run -p 8501:8501 --env-file .env langchain-search-agent
```
The app will be available on `http://localhost:8501`.
