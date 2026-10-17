from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
import os

@tool
def read_internal_strategy(doc_name: str) -> str:
    """
    Read internal strategy PDFs from the 'knowledge' folder. 
    Use this to find non-public strategic goals or internal project timelines.
    """
    try:
        # We look inside the 'knowledge' folder
        path = f"./knowledge/{doc_name}"
        if not os.path.exists(path):
            return f"Document {doc_name} not found in knowledge folder."
            
        loader = PyPDFLoader(path)
        pages = loader.load_and_split()
        # Combine the first few pages (to avoid overwhelming the AI)
        content = " ".join([p.page_content for p in pages[:3]])
        return content
    except Exception as e:
        return f"Error reading PDF: {e}"
    
    