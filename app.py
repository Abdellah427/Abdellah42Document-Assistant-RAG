
import streamlit as st

import src.interfaceG as interfaceG


def main():

    mistral_key= "5Lf75S6e7HwH2K4FDO2WViZVCTT0XSMH"

    # Initialize session state
    interfaceG.initialize_session_state()

    # Apply styles
    interfaceG.title()

    # Text input for user messages with action on Enter
    st.text_input("", value="", key="user_input", on_change=lambda: interfaceG.handle_send_message(mistral_key), placeholder="Enter your message here...")

    # Display the messages
    interfaceG.display_messages()

    # Handle file upload and database creation
    interfaceG.handle_file_upload()
    
    # Display documents retrieved
    interfaceG.display_documents()

if __name__ == "__main__":
    
    main()
