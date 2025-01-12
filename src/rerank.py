import faiss
import numpy as np
from mistralai import Mistral
import time
import pandas as pd
import re


def load_faiss_index(embeddings):
    embedding_matrix = np.array(embeddings)
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)
    return index

def get_embeddings_by_chunks(texts, client, model="mistral-embed", chunk_size=10, delay=1):
    chunks = [texts[x: x + chunk_size] for x in range(0, len(texts), chunk_size)]
    embeddings = []
    for i, chunk in enumerate(chunks):
        response = client.embeddings.create(model=model, inputs=chunk)
        embeddings.extend([d.embedding for d in response.data])
        break
        time.sleep(delay)
    return embeddings


def rerank_results(query, results, texts, client, model="mistral-large-latest"):
    ranked_results = []
    for idx in results:
        text = texts[idx]
        prompt = f"Query: {query}\nDocument: {text}\nRelevance (1-10):"
        response = client.chat.complete(model=model, messages=[{"role": "user", "content": prompt}])
        score = int(re.findall(r'\d+', response.choices[0].message.content.strip())[0])
        ranked_results.append((idx, score))
    return sorted(ranked_results, key=lambda x: x[1], reverse=True)


def search_and_rerank(query, client, index, texts, top_k=5):
    # Recherche initiale
    query_embedding = client.embeddings.create(model="mistral-embed", inputs=[query]).data[0].embedding
    distances, indices = index.search(np.array([query_embedding]), top_k)
    time.sleep(2)

    # Reranking
    ranked_indices = rerank_results(query, indices[0], texts, client)
    return [texts[idx] for idx in ranked_indices]