import streamlit as st
import os
import src.create_db as create_db
import src.helpers as helpers
import src.llm_interface as llm_interface

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

    # Additional page layout and styling adjustments
    st.markdown("""
        <style>
            .css-1d391kg { font-family: 'Arial', sans-serif; }
            .css-1r1ggxg { background-color: #f0f8ff; border-radius: 8px; padding: 10px; margin: 10px; }
            .css-ffhzg1 { font-size: 16px; }
        </style>
    """, unsafe_allow_html=True)

def create_box_choices(rag_methods, selected_method):
    """Function to create custom styled buttons for RAG methods."""
    
    st.markdown("""
        <style>
            
            .custom-radio-button {
                display: inline-block;
                background-color: #f0f8ff;
                border: 2px solid #007BFF;
                border-radius: 25px;
                padding: 10px 20px;
                margin: 5px;
                cursor: pointer;
                font-size: 16px;
                text-align: center;
                color: #007BFF;
                transition: background-color 0.3s, color 0.3s;
            }

            /* Apparence quand le bouton est sélectionné */
            .custom-radio-button.selected {
                background-color: #007BFF;
                color: white;
            }

            /* Aligner les boutons sur une ligne et les centrer */
            .radio-container {
                display: flex;
                justify-content: center;
                gap: 20px;
                margin: 10px 0;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="radio-container">', unsafe_allow_html=True)

    for method in rag_methods:
        if selected_method == method:
            st.markdown(f'''
                <div class="custom-radio-button selected" onclick="selectMethod('{method}')">
                    {method}
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
                <div class="custom-radio-button" onclick="selectMethod('{method}')">
                    {method}
                </div>
            ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # JavaScript 
    st.markdown("""
        <script>
            function selectMethod(method) {
                const buttons = document.querySelectorAll('.custom-radio-button');
                buttons.forEach(button => {
                    button.classList.remove('selected');
                });
                const selectedButton = document.querySelector(`.custom-radio-button:contains('${method}')`);
                selectedButton.classList.add('selected');

                window.parent.postMessage({ 'rag_method': method }, "*");
            }
        </script>
    """, unsafe_allow_html=True)



def initialize_session_state():
    """Initialize session state if necessary."""
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = ""
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'rag_method' not in st.session_state:
        st.session_state['rag_method'] = "Classic" 


def handle_send_message(mistral_key):
    """Handle sending a message from the user to the chatbot and update history."""
    user_input = st.session_state.user_input
    if user_input:
        # Preprocess user input
        processed_input = helpers.preprocess_input(user_input)

        # Query the most similar documents to the user input
        if st.session_state.rag_method == "Classic":
            docs = create_db.query_vector_db_colbertv2(user_input, 2)
        elif st.session_state.rag_method == "ColBERTv2":
            docs = create_db.query_vector_db_colbertv2(user_input, 2)
        elif st.session_state.rag_method == "Simon":
            docs = create_db.query_vector_db_colbertv2(user_input, 2)
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

    uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])

    if 'rag_method_locked' not in st.session_state:
        st.session_state['rag_method_locked'] = False

    if not st.session_state['rag_method_locked']:

        rag_methods = ["Classic", "ColBERTv2", "Simon"]

        create_box_choices(rag_methods, st.session_state.rag_method)

    else:
        st.write(f"RAG Method selected: **{st.session_state.rag_method}** (locked)")


    if st.button("Create Database"):
        if uploaded_files:
            csv_paths = []
            csv_folder = "uploaded_dataset"

            st.session_state['rag_method_locked'] = True
            os.makedirs(csv_folder, exist_ok=True)

            db_path = "database"
            os.makedirs(db_path, exist_ok=True)

            # Save the uploaded files
            for uploaded_file in uploaded_files:
                file_path = os.path.join(csv_folder, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                csv_paths.append(file_path)

            uploaded_file = uploaded_files[0]
            csv_path = os.path.join(csv_folder, uploaded_file.name)
            # Create the database based on the selected RAG method
            if st.session_state.rag_method == "Classic":
                create_db.create_vector_db_colbertv2(csv_path, db_path)
                st.success(f"Database created with Classic successfully!")
            elif st.session_state.rag_method == "ColBERTv2":
                create_db.create_vector_db_colbertv2(csv_path, db_path)
                st.success(f"Database created with ColBERTv2 successfully!")
            elif st.session_state.rag_method == "Simon":
                create_db.create_vector_db_colbertv2(csv_path, db_path)
                st.success(f"Database created with Simon successfully!")
        else:
            st.warning("Please upload CSV files.")

    elif st.session_state['rag_method_locked']:
        st.info("Database creation already in progress or completed. Reload the app to reset.")
