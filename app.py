import streamlit as st
import os
import logging
import faiss
import pandas as pd

import src.rerank as rerank
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

        response = Simon(user_input)

        formatted_response = helpers.format_response(response) # Pour l'instant on ne fait rien

        # Ajout de l'entrée de l'utilisateur et de la réponse du chatbot à l'historique
        st.session_state.history.append(f"You: {user_input}")
        st.session_state.history.append(f"Chatbot: {formatted_response}")

        # Réinitialisation de l'entrée utilisateur dans l'état de la session (pour effacer le champ de saisie)
        st.session_state.user_input = ""  # Clear input after sending


    # Champ de saisie pour les messages avec action sur Entrée
    st.text_input("You: ", value="", key="user_input", on_change=handle_send_message)

    # Affichage des messages
    for message in st.session_state.history:
        st.write(message)

    # Téléchargement de fichiers CSV
    uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=False, type=["csv"])

    if st.button("Create Database"):
        if uploaded_files:
            csv_folder = "uploaded_dataset"
            os.makedirs(csv_folder, exist_ok=True)

            # Enregistrer les fichiers téléchargés
            for uploaded_file in uploaded_files:
                with open(os.path.join(csv_folder, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())

            csv_path = os.path.join(csv_folder, uploaded_file.name)

            db_path = "./database"
            os.makedirs(db_path, exist_ok=True)
            uploaded_file = uploaded_files[0]
            csv_path = os.path.join(csv_folder, uploaded_file.name)
            db_path = os.path.join(db_path, uploaded_file.name)
            # Création de la base de données

        

            
        else:
            st.write("Please upload CSV files.")

if __name__ == "__main__":
    main()




def Simon(user_input):
    client, model = rerank.load_mistral()
    df = load_data("uploaded_dataset/wiki_movie_plots_deduped.csv").head(5000)
    
    # Chargement ou création des embeddings et de l'index FAISS au début
    if 'first_connexion' not in st.session_state:
        
        index, pca = rerank.create_vector_db_all_MiniLM_L6("uploaded_dataset/wiki_movie_plots_deduped.csv")
        # df = rerank.load_data("uploaded_dataset/wiki_movie_plots_deduped.csv").head(5000)
        # embedding_model = rerank.load_embedding_model()
        # embeddings = rerank.get_embeddings(df, embedding_model)
        # index, pca = rerank.load_faiss(embeddings)
        
        # Stockage des résultats dans st.session_state
        st.session_state['first_connexion'] = False
        # st.session_state['index'] = index
        # st.session_state['pca'] = pca
        # st.session_state['df'] = df
        # st.session_state['embedding_model'] = embedding_model

    else:
        # Chargement des résultats depuis st.session_state
        # embeddings = st.session_state['embeddings']
        # index = st.session_state['index']
        # pca = st.session_state['pca']
        # df = st.session_state['df']
        # embedding_model = st.session_state['embedding_model']
        pca = faiss.read_VectorTransform("pca_file")
        index = faiss.read_index("faiss_index_file")


    if user_input:
        # Prétraitement de l'entrée utilisateur 
        processed_input = helpers.preprocess_input(user_input) # Mise en minuscule et suppression des espaces inutiles
        results= rerank.search_and_rerank(pca, client, processed_input, index, df['Plot'].tolist())
        response = rerank.generate_final_response(client, processed_input, results)
        return response



@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)