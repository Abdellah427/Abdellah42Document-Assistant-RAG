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


class CustomVectorRetriever(BaseRetriever):
    """
    Custom retriever for vector-based document retrieval using FAISS and Sentence Transformers.
    
    Attributes:
        embedding_function (callable): Function used to convert text to vectors.
        index (faiss.IndexFlatL2): FAISS index for vector searching.
        documents (List[Document]): List of documents stored as Document objects.
    """
    embedding_function: callable = Field(...)
    index: object = Field(...)
    documents: list = Field(...)

    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve the top 'k' relevant documents for a given query string.

        Args:
            query (str): Query string to search for.
            k (int): Number of top documents to retrieve.

        Returns:
            List[Document]: List of top 'k' relevant Document objects.
        """
        query_embedding = self.embedding_function([query])
        distances, indices = self.index.search(query_embedding, k)
        return [self.documents[i] for i in indices[0]]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Asynchronously retrieve relevant documents for a given query string.

        Args:
            query (str): Query string to search for.

        Returns:
            List[Document]: List of relevant Document objects.
        """
        return self.get_relevant_documents(query)
    
def create_vector_db_Classic(csv_path: str, other_options_if_needed) -> None:
    """
    This function performs embedding and indexing.
    
    Args:
        csv_path (str): Path to the CSV file containing the data to be indexed.
        other_options_if_needed: Any additional options required for the method.
    """
    # Charger les données CSV
    df = pd.read_csv(csv_path)
    df = df[['Release Year', 'Title', 'Genre', 'Plot']]  # Adapté selon votre CSV
    df = df.dropna(subset=['Plot'])

    # Charger le modèle de transformation pour obtenir les embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = [embedding_model.encode([plot])[0] for plot in df['Plot'].tolist()]
    embeddings = np.array(embeddings)

    # Créer un index FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Créer la liste des documents
    documents = [
        Document(page_content=plot, metadata={'Title': title, 'Release Year': year, 'Genre': genre})
        for plot, title, year, genre in zip(df['Plot'], df['Title'], df['Release Year'], df['Genre'])
    ]

    # Créer l'instance du retriever avec FAISS et Sentence Transformer
    retriever = CustomVectorRetriever(embedding_function=embedding_model.encode, index=index, documents=documents)
    
    # Enregistrer l'objet retriever dans la session Streamlit
    st.session_state.retriever = retriever
    st.write("Embeddings and vector database created successfully!")

def query_vector_db_Classic(query_text: str, n_results: int = 5) -> list[dict]:
    """
    This function queries the vector database.
    
    Args:
        query_text (str): The input text for the query.
        n_results (int): Number of results to return (default is 5).
    
    Returns:
        list[dict]: A list of dictionaries containing the results.
    """
    # Vérifier si le retriever est disponible dans la session
    if 'retriever' in st.session_state:
        retriever = st.session_state.retriever
        
        # Obtenir les documents pertinents pour la requête
        relevant_docs = retriever.get_relevant_documents(query_text, k=n_results)

        # Retourner les documents récupérés sous forme de dictionnaire
        results = [{
            'text': doc.page_content,
            'metadata': doc.metadata
        } for doc in relevant_docs]

        return results
    else:
        st.warning("Vector database not created. Please create the database first.")
        return []


if __name__ == "__main__":
    # Example usage placeholder
    print("Module loaded. Implement main logic or import functions as needed.")
