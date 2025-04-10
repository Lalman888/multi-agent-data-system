from typing import List

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings  # Updated import for embeddings
# from langchain_pinecone import Pinecone as LCPinecone  # LangChain integration with Pinecone
from langchain_pinecone import PineconeVectorStore as LCPinecone  # LangChain integration with Pinecone
from langchain.schema import Document

class VectorDBManager:
    """Manages the Pinecone vector database for document storage and retrieval."""
    
    def __init__(self, api_key: str, namespace: str = "document-data", region: str = "us-east-1"):
        self.api_key = api_key
        self.namespace = namespace
        self.index_name = "multi-agent-data"
        self.dimension = 1536  # OpenAI embedding dimension
        
        # Initialize embedding model
        self.embedding_model = OpenAIEmbeddings()
        
        # Instantiate Pinecone with the provided API key
        self.pc = Pinecone(api_key=self.api_key)
        
        # Check and create index if it doesn't exist
        self._initialize_index(region)
        
        # Connect to the index
        self.index = self.pc.Index(self.index_name)
        
    def _initialize_index(self, region: str):
        """Initialize Pinecone index if it doesn't exist."""
        # Prepare a serverless spec (adjust cloud/region as needed)
        spec = ServerlessSpec(cloud="aws", region=region)
        
        # Get existing index names (using the new instance API)
        existing_indexes = self.pc.list_indexes().names()
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=spec
            )
            print(f"Created new Pinecone index: {self.index_name}")
    
    def store_documents(self, documents: List[Document]):
        """Create embeddings and store documents in Pinecone."""
        # Initialize LangChain's Pinecone integration
        vectorstore = LCPinecone.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            index_name=self.index_name,
            namespace=self.namespace
        )
        print(f"Stored {len(documents)} document chunks in Pinecone")
        return vectorstore
    
    def retrieve_similar(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve similar documents based on a query."""
        vectorstore = LCPinecone(
            index_name=self.index_name,
            embedding=self.embedding_model,
            namespace=self.namespace
        )
        return vectorstore.similarity_search(query, k=k)
    
    def create_retriever(self):
        """Create a retriever for use with LangChain."""
        vectorstore = LCPinecone(
            index_name=self.index_name,
            embedding=self.embedding_model,
            namespace=self.namespace
        )
        return vectorstore.as_retriever()
