from chromadb.config import Settings
from chromadb import Client
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import os

def create_vector_db(db_path):
    # Initialize ChromaDB client with the new configuration
    client = Client()

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
    collection.add(vectors=[vector], metadatas=[metadata])

def query_vector_db(collection, query_vector, n_results=5):
    # Query the collection for the most similar vectors
    results = collection.query(query_vector=query_vector, n_results=n_results)
    return results

def delete_vector(collection, vector_id):
    # Delete a vector from the collection by its ID
    collection.delete(ids=[vector_id])

def process_pdfs(pdf_folder, collection):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Example model
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            vectors = model.encode([text])
            add_vector(collection, vectors[0], {"filename": pdf_file})

if __name__ == "__main__":
    db_path = "D:/VisualStudioProject/rag-chatbot/database"
    pdf_folder = "D:/VisualStudioProject/rag-chatbot/pdf_folder"
    collection = create_vector_db(db_path)
    process_pdfs(pdf_folder, collection)