import streamlit as st
import os
import logging

import src.create_db as create_db
import src.helpers as helpers
import src.retrieve_data as retrieve_data
import src.llm_interface as llm_interface

def main():
    st.title("RAG Chatbot")
    st.write("Welcome to the RAG Chatbot powered by Mistral AI 70B!")

    # Input pour l'utilisateur
    user_input = st.text_input("You: ", "")
    
    if st.button("Send"):
        if user_input:
            processed_input = helpers.preprocess_input(user_input)
            response = llm_interface.query_mistral(processed_input)
            formatted_response = helpers.format_response(response)
            st.write(f"Chatbot: {formatted_response}")
        else:
            st.write("Please enter a message.")
    
    # Téléchargement de fichiers CSV
    uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])
    
    if st.button("Create Database"):
        if uploaded_files:
            csv_folder = "uploaded_dataset"
            os.makedirs(csv_folder, exist_ok=True)

            # Enregistrer les fichiers téléchargés
            for uploaded_file in uploaded_files:
                with open(os.path.join(csv_folder, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            db_path = "./database"
            # Création de la base de données
            collection = create_db.create_vector_db(db_path)  # Assurez-vous que cette fonction renvoie un objet valide
            logging.info("Database created successfully!")
            create_db.process_csvs(csv_folder, collection)  # Assurez-vous que cette fonction est correctement implémentée
            logging.info("CSV files processed successfully!")
            st.write("Database created successfully!")
        else:
            st.write("Please upload CSV files.")
    
    # Note: Vous n'avez plus besoin de rappeler create_db.create_vector_db(db_path) à la fin ici
    # car il est déjà appelé plus tôt dans le bloc "Create Database"

if __name__ == "__main__":
    db_path = "./database"  # Il est toujours préférable de définir db_path ici pour une meilleure lisibilité
    main()
