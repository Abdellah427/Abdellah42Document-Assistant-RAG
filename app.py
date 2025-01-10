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


    user_input = st.text_input("You: ", "")
    
    if st.button("Send"):
        if user_input:
            processed_input = helpers.preprocess_input(user_input)
            response = llm_interface.query_mistral(processed_input)
            formatted_response = helpers.format_response(response)
            st.write(f"Chatbot: {formatted_response}")
        else:
            st.write("Please enter a message.")
    
    uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])
    
    if st.button("Create Database"):
        if uploaded_files:
            csv_folder = "uploaded_dataset"
            os.makedirs(csv_folder, exist_ok=True)
            for uploaded_file in uploaded_files:
                with open(os.path.join(csv_folder, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            db_path = "./database"
            collection = create_db.create_vector_db(db_path)
            logging.info("Database created successfully!")
            create_db.process_csvs(csv_folder, collection)
            logging.info("CSV files processed successfully!")
            st.write("Database created successfully!")
        else:
            st.write("Please upload CSV files.")
    create_db.create_vector_db(db_path)
    


if __name__ == "__main__":
    db_path = "./database" 
    main()