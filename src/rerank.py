import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from mistralai import Mistral
import re
import time
import os
import logging

def load_mistral():
    api_key = "uvPKnZ4G0YFoM6KBIUkgF0KzE8dpmsgb"
    model = "mistral-embed"
    client = Mistral(api_key=api_key)
    return client, model

def load_embedding_model():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Plot"])
    return df


def get_embeddings(df,embedding_model):
    
    # Générer des embeddings pour les résumés
    embeddings = embedding_model.encode(df["Plot"].tolist(), show_progress_bar=True)

    return embeddings


def load_faiss(embeddings):
    # Réduction de la dimensionnalité avec PCA pour accélérer les recherches
    dimension = 128  # Réduire la dimension des embeddings
    pca = faiss.PCAMatrix(embeddings.shape[1], dimension)
    pca.train(embeddings)
    reduced_embeddings = pca.apply_py(embeddings)
    index = faiss.IndexFlatIP(dimension)  # Produit scalaire pour la similarité cosine
    index = faiss.IndexIVFFlat(index, dimension, 100)  # Clustering pour des recherches plus rapides
    index.train(reduced_embeddings)
    index.add(reduced_embeddings)
    return index, pca


def rerank_results(client, query, results, texts, model="mistral-large-latest"):
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


def search_and_rerank(pca,client, embedding_model, query, index, texts, top_k=3):
    # Recherche initiale
    query_embedding = embedding_model.encode([query])
    query_reduced = pca.apply_py(query_embedding)
    distances, indices = index.search(query_reduced, top_k)
    

    # Reranking
    ranked_indices = rerank_results(client, query, indices[0], texts)
    return [texts[idx] for idx in ranked_indices]


def generate_final_response(client, query, retrieved_texts, model="mistral-large-latest"):
    """
    Génère une réponse basée sur la requête et les textes récupérés.
    """
    context = "\n\n".join(retrieved_texts)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer(in less than 100 words):"

    # Appel au modèle de génération
    response = client.chat.complete(model=model, messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content.strip()


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

