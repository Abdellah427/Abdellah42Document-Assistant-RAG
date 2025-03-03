import streamlit as st
import os
import src.create_db as create_db
import src.helpers as helpers
import src.llm_interface as llm_interface
import src.rerank as rerank
import faiss

csv_pathGlobal = []

def title():
    """
    Function to apply all the styles and layout configurations for the RAG Chatbot.
    """
    # Set up a nice title and header

    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

    # Title and description with style
    st.markdown("""
        <h1 style="text-align: center; color: #007BFF;">RAG Chatbot</h1>
        <h3 style="text-align: center; color: #555;">Welcome to the Chatbot powered by AI !</h3>
        <p style="text-align: center; font-size: 16px; color: #777;">Interact with the AI and get insightful answers to your queries</p>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state if necessary."""
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = ""
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'rag_method' not in st.session_state:
        st.session_state['rag_method'] = "" 


def handle_send_message(mistral_key):
    """Handle sending a message from the user to the chatbot and update history."""
    user_input = st.session_state.user_input
    if user_input:
        # Preprocess user input
        processed_input = helpers.preprocess_input(user_input)

        # Query the most similar documents to the user input
        if st.session_state.rag_method == "Retriever":
            docs = create_db.query_vector_db_CustomVectorRetriever(user_input, 5) 
        elif st.session_state.rag_method == "ColBERTv2":
            docs = create_db.query_vector_db_colbertv2(user_input, 2)
        elif st.session_state.rag_method == "Rerank":
             pca = faiss.read_VectorTransform("pca_file")
             index = faiss.read_index("faiss_index_file")
             global csv_pathGlobal
             if csv_pathGlobal != None:
                full_doc=create_db.files_to_list_str(csv_pathGlobal)
                docs = rerank.search_and_rerank(pca, user_input, index, full_doc, top_k=3)
             else:
                docs = []
        else:
            docs = []

        st.session_state['docs'] = docs
        processed_input = f"Question: \n\n{user_input} \n\nHere are some documents to answer the question: \n\n{docs}"

        # Send request to the LLM model
        response = llm_interface.query_mistral(processed_input, st.session_state.history, mistral_key)

        # Format the LLM response
        formatted_response = helpers.format_response(response)

        # Update history with user input and chatbot response
        st.session_state.history.append(f"You: {user_input}")
        st.session_state.history.append(f"Chatbot: \n\n{formatted_response}")

        # Clear user input field after sending
        st.session_state.user_input = "" 
        return docs

def display_messages():
    """Display messages from the history in the interface."""
    for message in st.session_state.history:
        st.write(message)

def display_documents():
    """Display the documents retrieved by the chatbot."""

    if st.session_state.get('docs') :
        st.write("Documents retrieved by the chatbot:")
        doc_labels = [f"Document {i+1}" for i in range(len(st.session_state['docs']))]
        selected_doc = st.selectbox(
            "Select a document to view:",
            options=doc_labels,
        )
        selected_doc_index = doc_labels.index(selected_doc)
        selected_doc = st.session_state['docs'][selected_doc_index]
        st.write("Selected Document:")
        st.write(selected_doc)
    else:
        # Show a message if no documents were retrieved
        if len(st.session_state.history) > 0 and "Chatbot:" in st.session_state.history[-1]:
            st.write("No documents retrieved by the chatbot for this interaction.")

def handle_file_upload():
    """Handle the uploading of CSV files and the creation of the database."""

    uploaded_files = st.file_uploader(
        "Upload CSV or PDF files", 
        accept_multiple_files=True, 
        type=["csv", "pdf"]
    )

    if 'rag_method_locked' not in st.session_state:
        st.session_state['rag_method_locked'] = False

    if not st.session_state['rag_method_locked']:
        # Create buttons for selecting RAG method
        rag_methods = ["Retriever", "ColBERTv2", "Rerank"]
        selected_rag_method = None

        # Create columns and display buttons
        cols = st.columns(len(rag_methods))
        
    if not st.session_state['rag_method_locked']:
        # Create columns for displaying buttons horizontally and center them
        rag_methods = ["Retriever", "ColBERTv2", "Rerank"]
        selected_rag_method = st.session_state.get('rag_method', None)

        # Create empty columns and align them to the center
        cols = st.columns(len(rag_methods))

        # Loop through rag methods and create buttons
        for i, method in enumerate(rag_methods):
            with cols[i]:
               
                is_selected = (method == selected_rag_method)
                
                # Add custom CSS to highlight the selected button
                button_style = f"""
                <style>
                    .highlighted-button {{
                        background-color: {'#4CAF50' if is_selected else 'transparent'};
                        color: {'white' if is_selected else 'black'};
                        border: 2px solid {'#4CAF50' if is_selected else '#ccc'};
                    }}
                </style>
                """
                st.markdown(button_style, unsafe_allow_html=True)
                
                if st.button(method, key=method, use_container_width=True):
                    selected_rag_method = method
                    st.session_state.rag_method = selected_rag_method
                    st.session_state['rag_method_locked'] = True
                    
                    
                


    else:
        st.write(f"RAG Method selected: **{st.session_state.rag_method}** (locked)")


    if st.button("Create Database"):
        if uploaded_files:

            # 1. Initialize the session state

            csv_paths = []
            csv_folder = "uploaded_dataset"

            st.session_state['rag_method_locked'] = True
            os.makedirs(csv_folder, exist_ok=True)

            db_path = "database"
            os.makedirs(db_path, exist_ok=True)


            # 2. Save the uploaded files

            
            csv_paths = []  # List to store file paths

            for uploaded_file in uploaded_files:
                # Get the file path
                file_path = os.path.join(csv_folder, uploaded_file.name)

                # Save the uploaded file
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Get the file extension
                file_extension = os.path.splitext(uploaded_file.name)[-1].lower()

                # Add the file path to the list
                csv_paths.append(file_path)


            global csv_pathGlobal 
            csv_pathGlobal = csv_paths


            # 3. Extract text from PDF or CSV file


            full_doc=create_db.files_to_list_str(csv_paths)


                    

            # 4. Create the database based on the selected RAG method


            if st.session_state.rag_method == "Retriever":
                create_db.create_vector_db_all_MiniLM_L6(full_doc)
                st.success(f"Database created with Retriever successfully!")
            elif st.session_state.rag_method == "ColBERTv2":
                create_db.create_vector_db_colbertv2(full_doc)
                st.success(f"Database created with ColBERTv2 successfully!")
            elif st.session_state.rag_method == "Rerank":
                rerank.create_vector_db_all_MiniLM_L6_VS(full_doc)
                st.success(f"Database created with Rerank successfully!")
        else:
            st.warning("Please upload CSV files.")

    elif st.session_state['rag_method_locked']:
        selected_method = st.session_state.get('rag_method', 'Not selected')  # Get the selected method or default to 'Not selected'
        st.info(f"Database creation with the '{selected_method}' method is locked. Please reload the page to change the method.")
#