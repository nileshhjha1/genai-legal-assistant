import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# Global embeddings instance to avoid reloading
_embeddings = None


def get_embeddings():
    """Get or create the embeddings model (singleton pattern)"""
    global _embeddings
    if _embeddings is None:
        print("Loading embeddings model...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✓ Embeddings model loaded")
    return _embeddings


class VectorStoreManager:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "indian-constitution-ipc")

        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")

        self.pc = Pinecone(api_key=self.pinecone_api_key)

    def create_vector_store(self, pdf_path):
        """Load PDF and create vector store in Pinecone"""
        print("Loading PDF document...")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")

        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
        except Exception as e:
            raise Exception(f"Failed to load PDF: {str(e)}")

        print(f"Loaded {len(documents)} pages")

        # Better text splitting for legal documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Slightly larger chunks for legal context
            chunk_overlap=150,  # More overlap for better context
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Better separation for legal text
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")

        # Create or connect to Pinecone index
        print("Creating Pinecone vector store...")
        try:
            # Check if index exists, create if not
            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # all-MiniLM-L6-v2 has 384 dimensions
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print(f"Created new Pinecone index: {self.index_name}")

            # Wait for index to be ready
            import time
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)

            # Create vector store
            vector_store = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                index_name=self.index_name
            )

            print("Pinecone vector store created successfully!")
            return vector_store

        except Exception as e:
            raise Exception(f"Failed to create vector store: {str(e)}")

    def get_vector_store(self):
        """Get existing vector store from Pinecone"""
        try:
            print("Loading Pinecone vector store...")

            # Check if index exists
            if self.index_name not in self.pc.list_indexes().names():
                raise Exception("No vector store found. Please create vector store first.")

            # Connect to existing index
            vector_store = PineconeVectorStore.from_existing_index(
                index_name=self.index_name,
                embedding=self.embeddings
            )

            print("Pinecone vector store loaded successfully!")
            return vector_store

        except Exception as e:
            raise Exception(f"Failed to get vector store: {str(e)}")

    def check_index_exists(self):
        """Check if vector store exists in Pinecone"""
        try:
            return self.index_name in self.pc.list_indexes().names()
        except Exception as e:
            print(f"Error checking index existence: {e}")
            return False

    def delete_vector_store(self):
        """Delete existing vector store from Pinecone"""
        try:
            if self.check_index_exists():
                self.pc.delete_index(self.index_name)
                print("Vector store deleted successfully!")
                return True
            else:
                print("No vector store found to delete.")
                return False
        except Exception as e:
            print(f"Error deleting vector store: {str(e)}")
            return False

    def search_documents(self, query, k=5):
        """Direct document search for debugging"""
        try:
            vector_store = self.get_vector_store()
            results = vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []