import streamlit as st
from vectorized_database.create_db import create_vector_db
from langchain_process.connect_mistral import query_mistral
from utils.helpers import preprocess_input, format_response

def main():
    st.title("RAG Chatbot")
    st.write("Welcome to the RAG Chatbot powered by Mistral AI 70B!")

    user_input = st.text_input("You: ", "")
    
    if st.button("Send"):
        if user_input:
            processed_input = preprocess_input(user_input)
            response = query_mistral(processed_input)
            formatted_response = format_response(response)
            st.write(f"Chatbot: {formatted_response}")
        else:
            st.write("Please enter a message.")

if __name__ == "__main__":
    db_path = "D:/VisualStudioProject/rag-chatbot/database" 
    create_vector_db(db_path)
    main()