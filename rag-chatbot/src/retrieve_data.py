from chromadb import Client

def retrieve_data(database, user_query):
    client = Client()
    db = client.set_database(database)

    # Retrieve relevant data from the database
    relevant_data = db.query(user_query)

    return relevant_data