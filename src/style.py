import streamlit as st
import os

def apply_styles():
    """
    Function to apply all the styles and layout configurations for the RAG Chatbot.
    """
    # Set up a nice title and header
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

    # Title and description with style
    st.markdown("""
        <h1 style="text-align: center; color: #007BFF;">RAG Chatbot</h1>
        <h3 style="text-align: center; color: #555;">Welcome to the Chatbot powered by AI!</h3>
        <p style="text-align: center; font-size: 16px; color: #777;">Interact with the AI and get insightful answers to your queries.</p>
    """, unsafe_allow_html=True)

    # Additional page layout and styling adjustments
    st.markdown("""
        <style>
            .css-1d391kg { font-family: 'Arial', sans-serif; }
            .css-1r1ggxg { background-color: #f0f8ff; border-radius: 8px; padding: 10px; margin: 10px; }
            .css-ffhzg1 { font-size: 16px; }
        </style>
    """, unsafe_allow_html=True)
