import streamlit as st

def apply_styles():
    """
    Function to apply all the styles and layout configurations for the RAG Chatbot.
    """
    # Set up a nice title and header
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

    # Title and description with style
    st.markdown("""
        <h1 style="text-align: center; color: #4CAF50;">RAG Chatbot</h1>
        <h3 style="text-align: center; color: #888;">Welcome to the RAG Chatbot powered by Mistral AI!</h3>
        <p style="text-align: center; font-size: 16px; color: #555;">Interact with the AI and get answers powered by the latest Mistral model.</p>
    """, unsafe_allow_html=True)

    # Add a small description below with your school logo
    st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <img src="img/logo.png" alt="CY Tech Logo" width="300"/>
        </div>
    """, unsafe_allow_html=True)

    # Add a separator line to break content sections
    st.markdown("<hr>", unsafe_allow_html=True)
