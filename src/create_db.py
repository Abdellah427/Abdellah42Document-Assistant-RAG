from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import numpy as np
import uuid
from src.retrieve_data import retrieve_data
from ragatouille import RAGPretrainedModel
import shutil



# Abdellah


RAG_Corbert = None
def load_model_colbert():
    global RAG_Corbert

    # Vérifier si CUDA est disponible
    
    RAG_Corbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")



def csv_to_list_str(csv_path):
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
        # Joindre les colonnes pour créer une chaîne et l'ajouter à la liste
        text_output.append(" ".join(row_text))

    return text_output



# Fonction pour créer et sauvegarder l'index vectoriel avec ColBERTv2
def create_vector_db_colbertv2(csv_path, db_path):

    RAG_Corbert=RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    
    
    # Convertir le CSV en texte long
    liste = csv_to_list_str(csv_path)

    index_path=RAG_Corbert.index(
        collection=liste,  
        max_document_length=100,  
        split_documents=True,  
        use_faiss=True,
    )
    return index_path
    # Récupérer le nom du fichier sans l'extension
    file_name_without_ext = os.path.splitext(os.path.basename(csv_path))[0]
    index_name = file_name_without_ext + "_colbertv2"

    


    # Sauvegarder l'index dans un fichier

    #fichier_source = index_path
    #destination_path = db_path+"/"+index_name


    #if os.path.exists(destination_path) and os.path.isdir(destination_path):
    #    shutil.rmtree(destination_path)

    

    #shutil.move(fichier_source, destination_path)

    #RAG_Corbert=RAGPretrainedModel.from_index(destination_path, n_gpu=-1, verbose=1)

    return index_path


def query_vector_db_colbertv2(query_text, n_results=5):
    global RAG_Corbert
    # Charger le modèle pré-entraîné ColBERTv2
    if RAG_Corbert is None:
        load_model_colbert()

    # Rechercher les documents les plus similaires
    results = RAG_Corbert.search(query_text, n_results=n_results)

    return results

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
    print()
    
    #collection = create_vector_db(db_path)
    #process_csvs(csv_folder, collection)