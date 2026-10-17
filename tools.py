"""
tools.py
--------
Utility functions for text processing, TF-IDF retrieval, dataframe profiling,
and creating pandas query tools for the market intelligence agent.
"""

import re
from collections import Counter
from typing import List, Optional, Any, Callable
from dataclasses import dataclass
import pandas as pd
from langchain_core.tools import tool
import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())

# ---------------------------------------------------------------------------
# Text chunking utility
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks for retrieval.
    
    Args:
        text: The text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
    
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks


# ---------------------------------------------------------------------------
# TF-IDF Retriever
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk with its ID and text."""
    chunk_id: int
    text: str
    score: float = 0.0


class TfidfRetriever:
    """
    Simple keyword-based retriever for ranking chunks by relevance.
    Uses basic word matching instead of TF-IDF for simplicity.
    """
    
    def __init__(self, chunks: List[str]):
        """
        Initialize the retriever with chunks.
        
        Args:
            chunks: List of text chunks to index
        """
        self.chunks = chunks
    
    def retrieve(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        """
        Retrieve the top k most relevant chunks for a query.
        
        Args:
            query: Query string
            k: Number of chunks to retrieve
        
        Returns:
            List of RetrievedChunk objects sorted by relevance
        """
        if not self.chunks:
            return []
        
        # Simple keyword matching
        query_words = set(query.lower().split())
        
        scored_chunks = []
        for idx, chunk in enumerate(self.chunks):
            chunk_words = set(chunk.lower().split())
            # Calculate overlap score
            overlap = len(query_words & chunk_words)
            if overlap > 0:
                score = overlap / len(query_words)
                scored_chunks.append((idx, score))
        
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k
        results = [
            RetrievedChunk(
                chunk_id=idx,
                text=self.chunks[idx],
                score=score
            )
            for idx, score in scored_chunks[:min(k, len(scored_chunks))]
        ]
        
        return results

# ---------------------------------------------------------------------------
# Dataframe profiling
# ---------------------------------------------------------------------------

def dataframe_profile(df: pd.DataFrame) -> str:
    """
    Generate a simple profile/summary of a dataframe.
    
    Args:
        df: Pandas dataframe to profile
    
    Returns:
        String summary of the dataframe
    """
    if df.empty:
        return "Empty dataframe"
    
    profile = f"""
Dataframe Profile:
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {', '.join(df.columns)}
- Data types:
"""
    for col in df.columns:
        profile += f"  - {col}: {df[col].dtype}\n"
    
    profile += f"- Missing values: {df.isnull().sum().sum()}\n"
    profile += f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n"
    
    # Add basic stats for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        profile += "\n- Numeric columns summary:\n"
        for col in numeric_cols:
            profile += f"  - {col}: mean={df[col].mean():.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}\n"
    
    return profile


# ---------------------------------------------------------------------------
# Dataframe tool factory
# ---------------------------------------------------------------------------

def make_dataframe_tool(df: pd.DataFrame) -> Callable:
    """
    Create a langchain tool that allows querying a pandas dataframe using pandas/SQL syntax.
    
    Args:
        df: The pandas dataframe to query
    
    Returns:
        A callable tool that accepts queries and returns results
    """
    
    @tool
    def query_uploaded_csv(query: str) -> str:
        """
        Query the uploaded CSV file using pandas syntax or simple SQL-like queries.
        
        Args:
            query: A query string describing what data you want to retrieve
                   (e.g., "show all rows where revenue > 1000" or "get product_name and revenue")
        
        Returns:
            String representation of the query results
        """
        try:
            # Try to execute as a pandas query first
            # Simple pattern matching for common queries
            query_lower = query.lower()
            
            # Handle "show all" or generic queries
            if "show all" in query_lower or "all rows" in query_lower:
                return f"All rows from CSV:\n{df.to_string()}"
            
            # Handle column selection
            if "select" in query_lower or "get" in query_lower or "columns" in query_lower:
                words = query.split()
                potential_cols = [col for col in df.columns if col.lower() in query_lower]
                if potential_cols:
                    return f"Columns: {', '.join(potential_cols)}\n{df[potential_cols].to_string()}"
            
            # Handle filtering
            if "where" in query_lower or "filter" in query_lower or ">" in query or "<" in query:
                # Extract simple conditions like "revenue > 1000"
                try:
                    # Use df.query for simple pandas queries
                    result = df.query(query.split("where")[-1].strip() if "where" in query_lower else query)
                    return f"Query result:\n{result.to_string()}"
                except Exception:
                    pass
            
            # Default: return summary stats
            return f"CSV Preview:\n{df.head().to_string()}\n\nShape: {df.shape}\nColumns: {list(df.columns)}"
        
        except Exception as e:
            return f"Error querying CSV: {str(e)}"
    
    return query_uploaded_csv
