from langchain import OpenAI
from chromadb import Client
def query_mistral(user_query):

    # Initialize ChromaDB client
    client = Client()
    db = client.set_database("database")

    # Retrieve relevant data from the database
    relevant_data = db.query(user_query)

    # Initialize Mistral AI model
    model = OpenAI(model_name="mistral-70b")

    # Process the query with the Mistral AI model
    response = model.generate(relevant_data)

    return response