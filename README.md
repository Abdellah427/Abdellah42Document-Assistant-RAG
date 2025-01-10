# RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot using Streamlit for the user interface, ChromaDB for vector database management, and Langchain to connect to the Mistral AI 70B model for query processing.

## Project Structure

```
rag-chatbot
├── src
│   ├── app.py               # Main entry point for the Streamlit application
│   ├── chromadb
│   │   └── create_db.py     # Functions to create and manage the ChromaDB vector database
│   ├── langchain
│   │   └── connect_mistral.py # Connects ChromaDB with the Mistral AI model
│   └── utils
│       └── helpers.py       # Utility functions for input processing and response formatting
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd rag-chatbot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create the ChromaDB vector database by running:
   ```
   python src/chromadb/create_db.py
   ```

4. Start the Streamlit application:
   ```
   streamlit run src/app.py
   ```

## Usage Guidelines

- Once the application is running, you can interact with the chatbot through the Streamlit interface.
- Input your queries in the provided text box, and the chatbot will respond based on the information stored in the ChromaDB and processed by the Mistral AI model.

## Overview of Functionality

The RAG chatbot leverages a combination of retrieval and generation techniques to provide informative responses. The ChromaDB stores relevant vectors, while the Langchain integration allows for seamless querying of the Mistral AI model, ensuring accurate and contextually relevant answers.