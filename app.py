import streamlit as st
import os
import logging

import src.create_db as create_db
import src.helpers as helpers
import src.llm_interface as llm_interface

def main():
    st.title("RAG Chatbot")
    st.write("Welcome to the RAG Chatbot powered by Mistral AI !")
    api_key = "5Lf75S6e7HwH2K4FDO2WViZVCTT0XSMH"

    # Initialisation de l'état de session si nécessaire
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = ""
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Création d'une fonction pour gérer l'envoi des messages
    def handle_send_message():
        user_input = st.session_state.user_input
        if user_input:
            processed_input = helpers.preprocess_input(user_input)
            response = llm_interface.query_mistral(processed_input, st.session_state.history, api_key)
            formatted_response = helpers.format_response(response)
            st.session_state.history.append(f"You: {user_input}")
            st.session_state.history.append(f"Chatbot: {formatted_response}")
            st.session_state.user_input = ""  # Clear input after sending

    # Champ de saisie pour les messages avec action sur Entrée
    st.text_input("You: ", value="", key="user_input", on_change=handle_send_message)

    # Affichage des messages
    for message in st.session_state.history:
        st.write(message)

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
            collection = create_db.create_vector_db(db_path)
            logging.info("Database created successfully!")
            create_db.process_csvs(csv_folder, collection)
            logging.info("CSV files processed successfully!")
            st.write("Database created successfully!")
        else:
            st.write("Please upload CSV files.")

if __name__ == "__main__":
    main()
