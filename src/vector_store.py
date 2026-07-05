import os
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


def get_faiss_index(docs, index_path="./index/faiss_index", embed_model="models/embedding-001") -> FAISS:
    """Create or load a FAISS index from documents.

    Args:
        docs: List of LangChain Document objects.
        index_path: Directory where FAISS index will be saved/loaded.
        embed_model: Gemini model name for embeddings (e.g., models/embedding-001).
    Returns:
        FAISS vector store ready for similarity search.
    """
    # Ensure the index directory exists
    index_dir = Path(index_path)
    index_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("GEMINI_API_KEY")

    # Use Google Gemini embeddings from langchain_google_genai
    embeddings = GoogleGenerativeAIEmbeddings(model=embed_model, google_api_key=api_key)

    # If index files exist, load them; otherwise, create new index
    if (index_dir / "index.faiss").exists() and (index_dir / "docstore.pkl").exists():
        # FAISS in newer versions requires allow_dangerous_deserialization=True to load locally saved index
        vector_store = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
    else:
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(str(index_dir))
    return vector_store
