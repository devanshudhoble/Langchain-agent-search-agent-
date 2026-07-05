import os
from typing import List

from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from src.prompt_templates import SYSTEM_PROMPT, HUMAN_PROMPT


class SearchAgent:
    """
    LangChain search agent that:
    1. Uses a Gemini LLM (ChatGoogleGenerativeAI).
    2. Retrieves relevant passages from a FAISS vector store.
    3. Returns a natural-language answer.
    """

    def __init__(self, vector_store, llm_cfg: dict):
        """
        Args:
            vector_store: FAISS instance (already built or loaded).
            llm_cfg: Dictionary from config.yaml with keys `provider`,
                     `model` and `api_key`.
        """
        api_key = llm_cfg.get("api_key") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not found. Set env GEMINI_API_KEY.")

        # Initialize the Gemini chat model
        self.llm = ChatGoogleGenerativeAI(
            model=llm_cfg.get("model", "gemini-1.5-flash"),
            temperature=0.0,
            google_api_key=api_key,
        )

        # Build the retrieval chain
        # Using ConversationalRetrievalChain from langchain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate(
                    template=HUMAN_PROMPT,
                    input_variables=["context", "question"]
                )
            },
            return_source_documents=True,
        )

        # Simple chat history for the chain (list of (human, ai) tuples)
        self.chat_history: List[tuple] = []

    def run(self, query: str) -> str:
        """Run a single query and return the LLM answer."""
        result = self.chain({"question": query, "chat_history": self.chat_history})
        # Update chat history for next turn
        self.chat_history.append((query, result["answer"]))
        return result["answer"]
