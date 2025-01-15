import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from mistralai import Mistral
import re
import src.create_db as create_db

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
    dimension = min(dimension, embeddings.shape[0])
    
    pca = faiss.PCAMatrix(embeddings.shape[1], dimension)
    pca.train(embeddings)
    reduced_embeddings = pca.apply_py(embeddings)
    index = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(index, dimension, 10)
    index.train(reduced_embeddings)
    index.add(reduced_embeddings)
    return index, pca



def create_vector_db_all_MiniLM_L6_VS(data_liste) -> None:
    """
    This function performs embedding and indexing.
    
    Args:
        csv_path (str): Path to the CSV file containing the data to be indexed.
        other_options_if_needed: Any additional options required for the method.
    """
    
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(data_liste, show_progress_bar=True)
    index,pca= load_faiss(embeddings)
    faiss.write_index(index, "faiss_index_file")
    faiss.write_VectorTransform(pca, "pca_file")
    return index, pca



def rerank_results(query, results, texts, model="mistral-large-latest"):
    client, _ = load_mistral()
    prompts = [
        f"Query: {query}\nDocument: {texts[idx]}\nRelevance (1-10):"
        for idx in results
    ]
    responses = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt} for prompt in prompts]
    )

    ranked_results = []
    for idx, response in zip(results, responses.choices):
        score = int(re.findall(r'\d+', response.message.content.strip())[0])
        ranked_results.append((idx, score))

    ranked_results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked_results]


def search_and_rerank(pca, query, index, texts, top_k=3):


    # Recherche initiale
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embedding_model.encode([query])
    query_reduced = pca.apply_py(query_embedding)
    distances, indices = index.search(query_reduced, top_k)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    query_embedding = embedding_model.encode([query])
    query_reduced = pca.apply_py(query_embedding)
    distances, indices = index.search(query_reduced, top_k)

    # Directly return the content + distance for each document
    results = []
    for i, idx in enumerate(indices[0]):
        doc_content = texts[idx]  # Get the content of the document
        doc_distance = distances[0][i]  # Get the corresponding distance
        doc_distance = round(doc_distance,3)
        results.append(f"Content: {doc_content}\n\n Distance: {doc_distance}")

    return results


    # Reranking
    #ranked_indices = rerank_results( query, indices[0], texts)
    #return [texts[idx] for idx in ranked_indices]




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

