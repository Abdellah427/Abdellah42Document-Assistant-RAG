from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import uuid
from src.retrieve_data import retrieve_data
import ragatouille


# Abdellah

#def create_vector_db_colbertv2(db_path):




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