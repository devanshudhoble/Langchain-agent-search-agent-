import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def load_documents(source_path: str, allowed_extensions: List[str] = ["pdf", "txt", "md"]) -> List[Document]:
    """Load all supported documents from a directory.

    Args:
        source_path: Path to folder containing documents.
        allowed_extensions: List of file extensions to consider.
    Returns:
        List of LangChain Document objects.
    """
    docs = []
    base_path = Path(source_path)
    if not base_path.is_dir():
        raise FileNotFoundError(f"Document source folder not found: {source_path}")

    for ext in allowed_extensions:
        for file_path in base_path.rglob(f"*.{ext}"):
            try:
                if ext == "pdf":
                    loader = PyPDFLoader(str(file_path))
                else:
                    loader = TextLoader(str(file_path))
                docs.extend(loader.load())
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
    return docs
