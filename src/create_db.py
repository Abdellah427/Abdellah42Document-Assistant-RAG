import pandas as pd
from ragatouille import RAGPretrainedModel

import faiss
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from typing import List
import streamlit as st
import numpy as np
from pydantic import Field
from langchain.schema import BaseRetriever
import src.CustomVectorRetriever as CustomVectorRetriever



# Global variable to store the RAG model instance
RAG_Corbert = None


def load_model_colbert() -> None:
    """
    Loads the ColBERT model into the global variable `RAG_Corbert`.
    """
    global RAG_Corbert
    RAG_Corbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")


def csv_to_list_str(csv_path: str) -> list[str]:
    """
    Converts a CSV file to a list of strings where each string represents a row.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        list[str]: A list of strings representing the rows in the CSV.
    """
    df = pd.read_csv(csv_path)
    text_output = []

    for _, row in df.iterrows():
        row_text = [f"{column}: {row[column]}" for column in df.columns]
        text_output.append(" ".join(row_text))

    return text_output


def create_vector_db_colbertv2(csv_path: str, max_document_length=100)-> str:
    """
    Creates a vector database using the ColBERT model from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.
        db_path (str): Path to store the created database.
        max_document_length (int): Maximum length of the document to index.

    Returns:
        str: Path to the created index.
    """
    global RAG_Corbert

    # Load the model if not already loaded
    RAG_Corbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")


    # Convert CSV data to a list of strings
    documents = csv_to_list_str(csv_path)

    # Index the collection using the ColBERT model
    index_path = RAG_Corbert.index(
        collection=documents,
        max_document_length=max_document_length,  # Truncate documents longer than 100 tokens
        split_documents=True,    # Automatically split documents if too large
        use_faiss=True           # Use FAISS for efficient vector search
    )

    return index_path



def query_vector_db_colbertv2(query_text: str, n_results: int = 5) -> list[dict]:
    """
    Queries the vector database using the ColBERT model.

    Args:
        query_text (str): The query string.
        n_results (int): Number of top results to return.

    Returns:
        list[dict]: A list of results, each containing the retrieved text and its score.
    """
    global RAG_Corbert

    # Load the model if not already loaded
    if RAG_Corbert is None:
        load_model_colbert()

    # Perform the search on the indexed database
    results = RAG_Corbert.search(query_text, k=n_results)

    return results

#Romain




if __name__ == "__main__":
    # Example usage placeholder
    print("Module loaded. Implement main logic or import functions as needed.")
