
Here’s the README introduction in a format you can copy and paste as code:

markdown
Copy code
# Document Assistant RAG

## Overview

The **Document Assistant RAG** project leverages the **Retrieval-Augmented Generation (RAG)** model to create an intelligent documentation assistant. This system is designed to augment traditional document retrieval by integrating external data sources, such as CSV or PDF files, and generating context-aware responses for users based on their queries.

Using advanced techniques in information retrieval and natural language processing (NLP), this assistant can process large volumes of unstructured data, query relevant documents, and generate comprehensive answers to user questions in real-time. It can be applied to various domains, including technical documentation, legal documents, research papers, or any other corpus of textual information.

## Key Features

- **Multiple Document Formats**: Supports uploading and processing of both **CSV** and **PDF** files for document retrieval.
- **RAG-based Query System**: Utilizes the **Retrieval-Augmented Generation (RAG)** approach, where an initial retrieval step fetches relevant documents, and a language model generates accurate answers based on the retrieved data.
- **Multiple Retrieval Methods**: The system supports different retrieval methods such as:
  - **Retriever**: Uses a custom vector database for document retrieval.
  - **ColBERTv2**: A more advanced retrieval model based on BERT-based representations.
  - **Rerank**: A method that re-ranks retrieved documents based on similarity to the query.
- **Flexible Interface**: The web-based interface built with **Streamlit** allows users to upload multiple documents simultaneously and interact with the assistant via a chat-like interface.

## Installation

To get started with the **Document Assistant RAG**, you can set up the project environment locally or test it directly on **Google Colab**.

### Option 1: Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/Abdellah427/Document-Assistant-RAG.git
   cd Document-Assistant-RAG
   ```
2. Install the required dependencies using pip :

   ```bash
   pip install -r requirements.txt

   ```
3. Launch the Streamlit interface :

   ```bash
   streamlit run app.py
   ```
4. Open the link provided in your terminal to interact with the assistant.


## Installation


Clone the repository using the following command in your terminal:

```bash
git clone https://github.com/Abdellah427/Document-Assistant-RAG.git
cd Document-Assistant-RAG
```

Build the Docker image from the `Dockerfile` :

```bash
docker build -t image_rag .
```

This will create a Docker image named image_rag. If you already have a pre-built Docker image, you can skip this step.

## Running the program

To run the application via Docker, follow these steps:

Launch the Docker container with the following command:

```bash
docker run -p 8501:8501 image_rag
```

This will map port 8501 of the container to port 8501 on your local machine.

Once the container is running, you can access the application via:

- **Locally** : Open your browser and go to `http://localhost:8501`.
- **From another device on the same network** : Use the external URL provided by Docker, for example: `http://<votre-ip-publique>:8501`.

The Streamlit application interface will then be accessible.

