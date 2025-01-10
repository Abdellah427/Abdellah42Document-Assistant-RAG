
# Document Assistant RAG

This project implements a documentation assistant based on the Retrieval-Augmented Generation (RAG) model. It is designed to provide augmented information from external data and generate relevant responses for users.

## Prerequisites

Before launching the application, make sure you have the following installed on your machine:

- **Docker**: It is required to run the application in a container

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
