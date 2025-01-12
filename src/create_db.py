from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import numpy as np
import uuid
from src.retrieve_data import retrieve_data
import ragatouille
from ragatouille import RAGPretrainedModel
import shutil
from mistralai import Mistral
import faiss
import time
import re


# Abdellah

def csv_to_long_text(csv_path):
    # Charger le fichier CSV
    df = pd.read_csv(csv_path)

    # Initialiser une liste pour stocker les lignes de texte
    text_output = []

    # Parcourir chaque ligne du DataFrame
    for index, row in df.iterrows():
        row_text = []
        for column in df.columns:
            # Ajouter chaque colonne avec sa valeur dans la forme "colonne: valeur"
            row_text.append(f"{column}: {row[column]}")
        # Joindre les éléments de la ligne et les ajouter au texte final
        text_output.append(" ".join(row_text))

    # Joindre toutes les lignes de texte pour obtenir le texte final
    final_text = "\n".join(text_output)

    return final_text

RAG_Corbert = None
def load_model_Colbert():
    global RAG_Corbert
    if RAG_Corbert is None:  # Charger le modèle uniquement s'il n'est pas encore chargé
        print("Chargement du modèle ColBERTv2...")
        RAG_Corbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        print("Modèle ColBERTv2 chargé avec succès !")
    return RAG_Corbert



# Fonction pour créer et sauvegarder l'index vectoriel avec ColBERTv2
def create_vector_db_colbertv2(csv_path, db_path):
    # Charger le modèle pré-entraîné ColBERTv2
    #if RAG_Corbert is None:
    #    load_model_Colbert()
    
    RAG_Corbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    # Convertir le CSV en texte long
    text = csv_to_long_text(csv_path)

    # Récupérer le nom du fichier sans l'extension
    index_name = os.path.splitext(os.path.basename(csv_path))[0]+ "_colbertv2"
    
    print(index_name)
    # Indexer le texte
    RAG_Corbert.index(
        collection=[text],  # Utiliser le texte généré à partir du CSV
        index_name=index_name,  # Nom de l'index
        max_document_length=400,  # Limite de longueur des documents
        split_documents=True,  # Fractionner les documents trop longs
    )

    # Sauvegarder l'index dans un fichier

    fichier_source = './ragatouille/colbert/indexes/'+index_name
    destination = db_path+index_name

    # Déplace le fichier vers le nouveau répertoire
    shutil.copy(fichier_source, destination)


    return index_name



# Romain

def create_vector_db(db_path):
    # Initialize ChromaDB client with the new configuration
    client = Client(Settings(persist_directory=db_path))

    # Check if the collection already exists
    collection_name = "chatbot_vectors"
    if collection_name in client.list_collections():
        collection = client.get_collection(collection_name)
    else:
        # Create a collection for storing vectors
        collection = client.create_collection(collection_name)

    return collection

def add_vector(collection, vector, metadata):
    # Add a vector to the collection with associated metadata
    vector_id = str(uuid.uuid4())
    collection.add(ids=[vector_id], embeddings=[vector], metadatas=[metadata])

def query_vector_db(collection, query_vector, n_results=5):
    # Query the collection for the most similar vectors
    results = collection.query(query_vector=query_vector, n_results=n_results)
    return results

def delete_vector(collection, vector_id):
    # Delete a vector from the collection by its ID
    collection.delete(ids=[vector_id])

def process_csvs(csv_folder, collection):
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    for csv_file in os.listdir(csv_folder):
        if csv_file.endswith('.csv'):
            csv_path = os.path.join(csv_folder, csv_file)
            df = pd.read_csv(csv_path)
            print(f"Processing file: {csv_file}")
            for index, row in df.iterrows():
                text = " ".join(row.astype(str).tolist())
                vectors = model.encode([text])
                add_vector(collection, vectors[0], {"filename": csv_file, "row_index": index})
                if (index + 1) % 1000 == 0:
                    print(f"Processed row {index + 1} of {len(df)} in file {csv_file}")

if __name__ == "__main__":
    db_path = "./database"
    csv_folder = "./uploaded_dataset"
    collection = create_vector_db(db_path)
    process_csvs(csv_folder, collection)



##Simon


# def save_embeddings_to_chroma(data, db_path="./data/chroma_db"):
#     chroma_client = Client(Settings(persist_directory=db_path))
#     collection = chroma_client.get_or_create_collection(name="movie_embeddings")
#     for idx, row in data.iterrows():
#         collection.add(
#             documents=[row['plot_synopsis']],
#             metadatas={"id": row.name},
#             embeddings=[row['embeddings']]
#         )
#     chroma_client.persist()

#     print("embeddings saved")

def get_and_save_embeddings_to_chroma(texts, client, db_path, model="mistral-embed", chunk_size=512, delay=1):

  # Create ChromaDB client
  chroma_client = Client(Settings(persist_directory=db_path))

  # Check if collection exists (optional)
  collection_name = "text_embeddings"
  if collection_name not in chroma_client.list_collections():
      collection = chroma_client.create_collection(collection_name)
  else:
      collection = chroma_client.get_collection(collection_name)

  # Process text chunks
  for i, chunk in enumerate(chunker(texts, chunk_size)):
    # Generate embeddings for the chunk
    embeddings_response = client.embeddings.create(model=model, inputs=chunk)
    embeddings = [d.embedding for d in embeddings_response.data]

    # Add embeddings and metadata to ChromaDB
    for j, text in enumerate(chunk):
      metadata = {"text": text}  # Add relevant metadata (e.g., text itself)
      collection.add(ids=[f"{i}_{j}"], embeddings=embeddings[j:j+1], metadatas=[metadata])

    print(f"Processed chunk {i+1} of {len(texts) // chunk_size + 1}")
    time.sleep(delay)

  chroma_client.persist()
  print("Embeddings saved successfully to ChromaDB!")
  return embeddings

# Helper function to create chunks (avoids index out of bounds)
def chunker(iterable, chunksize):
  """Yields successive n-sized chunks from an iterable."""
  for i in range(0, len(iterable), chunksize):
    yield iterable[i:i + chunksize]
