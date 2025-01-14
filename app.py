import streamlit as st
import os
import logging

import src.rerank as rerank
import src.create_db as create_db
import src.helpers as helpers
import src.llm_interface as llm_interface

def main():
    st.title("RAG Chatbot")
    st.write("Welcome to the RAG Chatbot powered by Mistral AI !")
    api_key = "5Lf75S6e7HwH2K4FDO2WViZVCTT0XSMH"

    #create_db.load_model_Colbert()
    INDEX_PATH = "saved_index/faiss_index"

    # Initialisation de l'état de session si nécessaire
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = ""
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Création d'une fonction pour gérer l'envoi des messages
    def handle_send_message():
        user_input = st.session_state.user_input
        if user_input:
            # Prétraitement de l'entrée utilisateur 
            processed_input = helpers.preprocess_input(user_input) # Mise en minuscule et suppression des espaces inutiles

            # ICI ON DOIT APPELER LA FONCTION les fonction qui ajoute les embeldings les plus proches à response


            ##test simon  rerank

            client, model = llm_interface.load_mistral()
            # texts = create_db.csv_to_long_text("uploaded_dataset/wiki_movie_plots_deduped.csv")
            # embeddings = rerank.get_embeddings_by_chunks(texts, client, model)
            # # embeddings = create_db.get_and_save_embeddings_to_chroma(texts, client,"./database")
            # index = rerank.load_faiss_index(embeddings)
            # results= rerank.search_and_rerank(processed_input, client, index, texts)
            # response = llm_interface.query_mistral(results, st.session_state.history, api_key) 
            # print(results)

            df = rerank.load_data("uploaded_dataset/wiki_movie_plots_deduped.csv").head(5000)
            embedding_model= rerank.load_embedding_model()
            embeddings = rerank.get_embeddings(df,embedding_model)
            index, pca = rerank.load_faiss(embeddings)
            results= rerank.search_and_rerank(pca, client, embedding_model, processed_input, index, df['Plot'].tolist())
            # response = llm_interface.query_mistral(results, st.session_state.history, api_key) 
            response = rerank.generate_final_response(client, processed_input, results)

            # Envoi de la requête au model LLM avec l'historique des échanges et la clé API
            # response = llm_interface.query_mistral(processed_input, st.session_state.history, api_key) 

            

            # Formatage de la réponse retournée par le LLM
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
    uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])

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

            #Celle de Romain
            
            # collection = create_db.create_vector_db(db_path)
            # logging.info("Database created successfully!")
            # create_db.process_csvs(csv_folder, collection)
            # logging.info("CSV files processed successfully!")
            
            # st.write("Database created successfully!")
            

            #Celle de ColBERTv2
            # index_name = create_db.create_vector_db_colbertv2(csv_path,db_path)
            


            ## mistral

            # client, model = llm_interface.load_mistral()
            # texts = create_db.csv_to_long_text("uploaded_dataset/wiki_movie_plots_deduped.csv")
            # # embeddings = rerank.get_embeddings_by_chunks(texts, client, model)
            # embeddings = create_db.get_and_save_embeddings_to_chroma(texts, client,"./database")

            
        else:
            st.write("Please upload CSV files.")

if __name__ == "__main__":
    main()
