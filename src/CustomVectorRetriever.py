from langchain.schema import Document
from pydantic import Field
from typing import List
from langchain.schema import BaseRetriever

class CustomVectorRetriever(BaseRetriever):
    """
    Custom retriever for vector-based document retrieval using FAISS and Sentence Transformers.
    
    Attributes:
        embedding_function (callable): Function used to convert text to vectors.
        index (faiss.IndexFlatL2): FAISS index for vector searching.
        documents (List[Document]): List of documents stored as Document objects.
    """
    from typing import Callable

    embedding_function: Callable = Field(...)

    index: object = Field(...)
    documents: list = Field(...)

    def _get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve the top 'k' relevant documents for a given query string.

        Args:
            query (str): Query string to search for.
            k (int): Number of top documents to retrieve.

        Returns:
            List[Document]: List of top 'k' relevant Document objects.
        """
        query_embedding = self.embedding_function([query])
        distances, indices = self.index.search(query_embedding, k)
        return [self.documents[i] for i in indices[0]]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Asynchronously retrieve relevant documents for a given query string.

        Args:
            query (str): Query string to search for.

        Returns:
            List[Document]: List of relevant Document objects.
        """
        return self._get_relevant_documents(query)
