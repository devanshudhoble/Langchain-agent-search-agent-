import os
from typing import List

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory


class SearchAgent:
    """
    AI Search Agent powered by Google Gemini and LangChain.
    Equipped with a DuckDuckGo web search tool to answer questions in real-time.
    """
    def __init__(self, llm_cfg: dict):
        api_key = llm_cfg.get("api_key")
        if not api_key or api_key == "YOUR_GEMINI_API_KEY":
            api_key = os.getenv("GEMINI_API_KEY")
            
        if not api_key:
            raise ValueError("Gemini API key not found. Set env GEMINI_API_KEY.")

        # Initialize the chat model with REST transport for robustness
        self.llm = ChatGoogleGenerativeAI(
            model=llm_cfg.get("model", "gemini-1.5-flash"),
            temperature=0.0,
            google_api_key=api_key,
            transport="rest"
        )

        # Initialize web search tool
        self.search_tool = DuckDuckGoSearchRun()
        self.tools = [self.search_tool]

        # Define prompt template for the agent
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful, smart AI search assistant. Use the web search tool to find accurate, real-time information when asked about news, events, or facts. Provide concise and informative answers."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create the tool-calling agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)

        # Create the AgentExecutor
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

        # In-memory history for conversation state
        self.chat_history = InMemoryChatMessageHistory()

    def run(self, query: str) -> str:
        """Run the agent with conversation memory."""
        # Convert history messages to a list
        history_msgs = self.chat_history.messages
        
        # Invoke executor
        result = self.executor.invoke({
            "input": query,
            "chat_history": history_msgs
        })
        
        # Add messages to history
        self.chat_history.add_user_message(query)
        self.chat_history.add_ai_message(result["output"])
        
        return result["output"]
