# Document Assistant RAG

## Overview

The **Document Assistant RAG** project leverages the **Retrieval-Augmented Generation (RAG)** model to create an intelligent documentation assistant. This system is designed to augment traditional document retrieval by integrating external data sources, such as CSV or PDF files, and generating context-aware responses for users based on their queries.

By combining advanced information retrieval and natural language processing (NLP) techniques, this assistant can process large datasets, retrieve relevant documents, and generate accurate, comprehensive answers in real-time. It is well-suited for domains such as:
- Technical documentation
- Legal texts
- Research papers
- Any corpus of unstructured textual data

## Key Features

- **Multiple Document Formats**: Supports uploading and processing of both **CSV** and **PDF** files for document retrieval.
- **RAG-based Query System**: Utilizes the **Retrieval-Augmented Generation (RAG)** approach, where an initial retrieval step fetches relevant documents, and a language model generates accurate answers based on the retrieved data.
- **Multiple Retrieval Methods**: The system supports different retrieval methods such as:
  - **Retriever**: Uses a custom vector database for document retrieval.
  - **ColBERTv2**: A more advanced retrieval model based on BERT-based representations.
  - **Rerank**: A method that re-ranks retrieved documents based on similarity to the query.
- **Interactive Web Interface**: The web-based interface built with **Streamlit** allows users to upload multiple documents simultaneously and interact with the assistant via a chat-like interface.



‎ 

## Installation

To get started with the **Document Assistant RAG**, you can either set it up locally or test it on **Google Colab**. Both methods are detailed below.

---


### Option 1: Test on Google Colab

If you want to quickly test the project without installing it locally, use the Google Colab notebook:


1. Open the Colab notebook by clicking the link below:  
   **[Test on Google Colab](https://colab.research.google.com/drive/1hMYZopKnL2nI3t8w-zQtZYZz5l-0XYIs?usp=sharing)**

2. **Select a T4 GPU** in the Colab runtime settings for optimal performance. This can be done by navigating to:
   - `Runtime` → `Change runtime type` → Set **Hardware accelerator** to **GPU** (preferably T4).

3. Follow the instructions in the notebook or consult the YouTube tutorial for guidance:  
   **[Watch Tutorial on YouTube](https://www.youtube.com/watch?v=-UWt5vqsP4Y)**

4. The notebook will guide you through the following steps:
   - **Step 1**: Upload your documents.
   - **Step 2**: Unzip the uploaded documents so you can use them for test.
   - **Step 3**: Select a retrieval method.
   - **Step 4**: Ask questions and interact with the assistant.

---


### Option 2: Run Locally

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

## Authors

This project was developed as part of the academic year **2024–2025** at **CY Tech**.

- **Abdellah Hassani**
- **Romain Guenneau**
- **Simon Ren**
- **Ritchy Agnesa**
