import os
from pathlib import Path
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai


class NativeGeminiEmbeddings(Embeddings):
    """Custom LangChain Embeddings wrapper using native google-generativeai SDK.
    
    This avoids gRPC connection issues/timeouts (504 Deadline Exceeded) 
    common with the standard langchain-google-genai gRPC transport behind 
    certain firewalls or local proxy settings, by using standard HTTPS REST calls.
    """
    def __init__(self, model: str = "models/gemini-embedding-001", google_api_key: str = None):
        self.model = model
        genai.configure(api_key=google_api_key)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            # Avoid sending blank/whitespace content which can fail
            if not text.strip():
                text = " "
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        if not text.strip():
            text = " "
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']


def get_faiss_index(docs, index_path="./index/faiss_index", embed_model="models/gemini-embedding-001") -> FAISS:
    """Create or load a FAISS index from documents.

    Args:
        docs: List of LangChain Document objects.
        index_path: Directory where FAISS index will be saved/loaded.
        embed_model: Gemini model name for embeddings.
    Returns:
        FAISS vector store ready for similarity search.
    """
    # Ensure the index directory exists
    index_dir = Path(index_path)
    index_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("GEMINI_API_KEY")

    # Use the robust custom NativeGeminiEmbeddings class
    embeddings = NativeGeminiEmbeddings(model=embed_model, google_api_key=api_key)

    # If index files exist, load them; otherwise, create new index
    if (index_dir / "index.faiss").exists() and (index_dir / "docstore.pkl").exists():
        vector_store = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
    else:
        # Create and save FAISS index
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(str(index_dir))
    return vector_store
