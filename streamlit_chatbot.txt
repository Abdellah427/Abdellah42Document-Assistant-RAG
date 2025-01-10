import streamlit as st
from transformers import pipeline

# Suppress warnings from transformers and torch
from transformers import logging as transformers_logging
import torch
transformers_logging.set_verbosity_error()
torch._C._log_api_usage_once = lambda *args, **kwargs: None

# Function to handle user input and generate responses using Hugging Face model
def chatbot_response(user_input):
    conversation_history = "\n".join(st.session_state.history[-5:])  # Use the last 5 messages for context
    prompt = f"{conversation_history}\nYou: {user_input}\nBot:"
    response = st.session_state.chatbot(
        prompt,
        max_length=150,  # Increase max_length for longer responses
        num_return_sequences=1,
        truncation=True,
        pad_token_id=50256,
        temperature=0.7,  # Control randomness
        top_p=0.9,  # Control diversity
        repetition_penalty=1.2,  # Penalize repeated tokens
        do_sample=True  # Enable sampling
    )
    return response[0]['generated_text'][len(prompt):].strip()

def main():
    # Initialize chat history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Initialize the Hugging Face model
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

    # Streamlit app layout
    st.title("Chatbot Interface")

    # User input form
    form = st.form(key='chat_form')
    user_input = form.text_input("You:", "")
    submit_button = form.form_submit_button(label='Send')

    # Handle form submission
    if submit_button and user_input:
        response = chatbot_response(user_input)
        st.session_state.history.append(f"You: {user_input}")
        st.session_state.history.append(f"Bot: {response}")

    # Display chat history
    for message in st.session_state.history:
        st.write(message)

if __name__ == "__main__":
    main()
