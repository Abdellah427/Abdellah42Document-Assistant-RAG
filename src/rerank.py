import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from mistralai import Mistral
import re

def load_mistral():
    """
    Initialize and load the Mistral client with the specified API key and model.

    Returns:
        Tuple[Mistral, str]: A tuple containing the initialized Mistral client 
        and the name of the model to be used.
    """
    api_key = "uvPKnZ4G0YFoM6KBIUkgF0KzE8dpmsgb"
    model = "mistral-embed"
    client = Mistral(api_key=api_key)
    return client, model


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Plot"])
    return df


def get_embeddings(df,embedding_model):
    
    # Générer des embeddings pour les résumés
    embeddings = embedding_model.encode(df["Plot"].tolist(), show_progress_bar=True)

    return embeddings


def load_faiss(embeddings: np.ndarray) :
    """
    Load and build a FAISS index with dimensionality reduction using PCA for faster similarity searches.
    Args:
        embeddings (np.ndarray): Array of original embeddings to be indexed.
    Returns:
        Tuple[faiss.IndexIVFFlat, faiss.PCAMatrix]: A tuple containing the trained FAISS index
        and the PCA matrix used for dimensionality reduction.
    """
    dimension = 128  
    pca = faiss.PCAMatrix(embeddings.shape[1], dimension)
    pca.train(embeddings)
    reduced_embeddings = pca.apply_py(embeddings)
    index = faiss.IndexFlatIP(dimension)  
    index = faiss.IndexIVFFlat(index, dimension, 100)  
    index.train(reduced_embeddings)
    index.add(reduced_embeddings)
    return index, pca


def create_vector_db_all_MiniLM_L6_VS(csv_path: str) -> None:
    """
    This function performs embedding and indexing.
    
    Args:
        csv_path (str): Path to the CSV file containing the data to be indexed.
        other_options_if_needed: Any additional options required for the method.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Plot"]).head(5000)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(df["Plot"].tolist(), show_progress_bar=True)
    index,pca= load_faiss(embeddings)
    faiss.write_index(index, "faiss_index_file")
    faiss.write_VectorTransform(pca, "pca_file")
    return index, pca



def rerank_results(query, results, texts, model="mistral-large-latest"):
    """
    Reranks the retrieved documents using a large language model (LLM).
    
    Args:
        query (str): The user's query.
        results (list[int]): List of indices of the initially retrieved documents.
        texts (list[str]): List of document contents.
        model (str): The LLM model to use for reranking (default is "mistral-large-latest").
    
    Returns:
        list[str]: The documents reordered by relevance based on the LLM's scores.
    """
    # Load the LLM client
    client, _ = load_mistral()
    
    # Create prompts for the LLM
    prompts = [
        f"Query: {query}\nDocument: {texts[idx]}\nRelevance (1-10):"
        for idx in results
    ]
    
    # Get responses from the LLM
    responses = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt} for prompt in prompts]
    )
    
    # Extract scores and rerank documents
    ranked_results = []
    for idx, response in zip(results, responses.choices):
        score = int(re.findall(r'\d+', response.message.content.strip())[0])
        ranked_results.append((idx, score))
    
    # Sort documents by their scores in descending order
    ranked_results.sort(key=lambda x: x[1], reverse=True)
    
    # Return the documents in the reranked order
    return [texts[r[0]] for r in ranked_results]

def search_and_rerank(pca, query, index, texts, top_k=3):
    """
    Recherche initiale et rerank des documents les plus pertinents.

    Args:
        pca: Objet PCA pour la réduction de dimension.
        query (str): Requête utilisateur.
        index: Index de recherche (e.g., FAISS).
        texts (list[str]): Liste des documents.
        top_k (int): Nombre de résultats à retourner.

    Returns:
        list[str]: Les documents rerankés avec leurs distances.
    """
    # Recherche initiale
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embedding_model.encode([query])
    query_reduced = pca.apply_py(query_embedding)
    distances, indices = index.search(query_reduced, top_k)

    # Reranking
    ranked_indices = rerank_results(query, indices[0], texts)

    # Construction des résultats rerankés
    ranked_results = []
    for idx in ranked_indices[:top_k]:  # Limite aux top_k résultats rerankés
        if idx in indices[0]:  # Vérifie si l'indice existe dans la recherche initiale
            original_index = indices[0].tolist().index(idx)  # Trouve l'indice original
            ranked_results.append(
                f"Document: {texts[idx]}, Distance: {distances[0][original_index]:.4f}"
            )
        else:
            # Gérer les cas où idx n'est pas dans indices[0]
            ranked_results.append(
                f"Document: {texts[idx]}, Distance: N/A (Not found in initial search)"
            )

    return ranked_results




import pandas as pd

def detect_summary_column(df):
    """
    Détecte automatiquement la colonne contenant les résumés.
    Stratégies utilisées :
    1. Recherche de mots-clés dans les noms de colonnes.
    2. Détection par la longueur moyenne des textes.
    """
    # Liste de mots-clés pour les noms de colonnes
    keywords = ['summary', 'plot', 'description', 'text', 'content']
    
    # Étape 1 : Chercher une colonne avec un mot-clé dans son nom
    for col in df.columns:
        if any(keyword in col.lower() for keyword in keywords):
            return col
    
    # Étape 2 : Si aucun mot-clé trouvé, utiliser la colonne avec la plus grande longueur moyenne
    avg_lengths = df.apply(lambda col: col.astype(str).str.len().mean())
    return avg_lengths.idxmax()

