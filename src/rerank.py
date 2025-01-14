# import faiss
# import numpy as np
# from mistralai import Mistral
# import time
# import pandas as pd
# import re
# from sentence_transformers import SentenceTransformer

# def load_faiss_index(embeddings):
#     embedding_matrix = np.array(embeddings)
#     dimension = embedding_matrix.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(embedding_matrix)
#     return index

# def get_embeddings_by_chunks(texts, client, model="mistral-embed", chunk_size=1024, delay=3):
#     chunks = [texts[x: x + chunk_size] for x in range(0, len(texts), chunk_size)]
#     embeddings = []
#     for i, chunk in enumerate(chunks):
#         response = client.embeddings.create(model=model, inputs=chunk)
#         embeddings.extend([d.embedding for d in response.data])
#         break
#         time.sleep(delay)
#     return embeddings


# def rerank_results(query, results, texts, client, model="mistral-large-latest"):
#     ranked_results = []
#     for idx in results:
#         text = texts[idx]
#         prompt = f"Query: {query}\nDocument: {text}\nRelevance (1-10):"
#         response = client.chat.complete(model=model, messages=[{"role": "user", "content": prompt}])
#         score = int(re.findall(r'\d+', response.choices[0].message.content.strip())[0])
#         ranked_results.append((idx, score))
#     return sorted(ranked_results, key=lambda x: x[1], reverse=True)


# def search_and_rerank(query, client, index, texts, top_k=5):
#     # Recherche initiale
#     logging.info(f"Starting search and reranking for query: {query}")
#     query_embedding = client.embeddings.create(model="mistral-embed", inputs=[query]).data[0].embedding
#     distances, indices = index.search(np.array([query_embedding]), top_k)
#     time.sleep(5)

#     # Reranking
#     ranked_indices = rerank_results(query, indices[0], texts, client)
#     print(ranked_indices)
#     return [texts[idx] for idx, _ in ranked_indices]

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from mistralai import Mistral
import re
import time
import os
import logging

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

# def load_or_create_index(df, INDEX_PATH, embedding_model):
#     if os.path.exists(INDEX_PATH):
#         index = faiss.read_index(INDEX_PATH)
#         logging.info("Index loaded from disk.")
#     else:
#         embeddings=get_embeddings(df, embedding_model)
#         dimension = 128  # Réduire la dimension des embeddings
#         pca = faiss.PCAMatrix(embeddings.shape[1], dimension)
#         pca.train(embeddings)
#         reduced_embeddings = pca.apply_py(embeddings)
#         index = faiss.IndexFlatIP(dimension)  # Produit scalaire pour la similarité cosine
#         index = faiss.IndexIVFFlat(index, dimension, 100)  # Clustering pour des recherches plus rapides
#         index.train(reduced_embeddings)
#         index.add(reduced_embeddings)
#         return index, pca

def rerank_results(client, query, results, texts, model="mistral-large-latest"):
    ranked_results = []
    for idx in results:
        text = texts[idx]
        prompt = f"Query: {query}\nDocument: {text}\nRelevance (1-10):"
        response = client.chat.complete(model=model, messages=[{"role": "user", "content": prompt}], max_tokens=300 )
        score = int(re.findall(r'\d+', response.choices[0].message.content.strip())[0])
        ranked_results.append((idx, score))

    # Trier par pertinence décroissante
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

# Exemple d'utilisation
# df = pd.read_csv("\uploaded_dataset\wiki_movie_plots_deduped.csv")
# summary_column = detect_summary_column(df)
# print(f"Colonne détectée pour les résumés : {summary_column}")
# resumes = df[summary_column].tolist()
