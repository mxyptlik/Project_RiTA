import logging
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from chromadb.config import Settings

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaService:
    _instance = None
    _retriever = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChromaService, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        """Initializes the ChromaDB vector store and retriever."""
        if self._retriever:
            logger.info("ChromaDB service already initialized.")
            return

        try:
            embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"))
            
            environment = os.getenv("ENVIRONMENT", "local").strip()
            logger.info(f"Running in {environment} environment.")

            if environment == 'local':
                # Local setup: Connect to a persistent directory
                vector_store = Chroma(
                    persist_directory=os.getenv("CHROMA_DB_PATH", "./chroma_db"),
                    embedding_function=embeddings,
                    collection_name="bella_document"
                )
            else:
                # Docker/Server setup: Connect to ChromaDB server via HTTP
                chroma_settings = Settings(
                    chroma_api_impl="chromadb.api.fastapi.FastAPI",
                    chroma_server_host=os.getenv("CHROMA_SERVER_HOST", "localhost"),
                    chroma_server_http_port=os.getenv("CHROMA_SERVER_HTTP_PORT", "8000")
                )
                vector_store = Chroma(
                    client_settings=chroma_settings,
                    embedding_function=embeddings,
                    collection_name="bella_document"
                )

            if vector_store._collection.count() == 0:
                logger.warning("ChromaDB collection is empty. Please run ingest.py to add documents.")
            else:
                logger.info(f"ChromaDB contains {vector_store._collection.count()} documents.")

            self._retriever = vector_store.as_retriever(search_kwargs={"k": int(os.getenv("RETRIEVER_K", "10"))})
            logger.info("✅ ChromaDB retriever initialized successfully.")

        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB retriever: {e}", exc_info=True)
            self._retriever = None

    async def retrieve_documents(self, query: str):
        """Retrieves relevant documents from ChromaDB based on a query."""
        if not self._retriever:
            logger.error("Retriever is not initialized.")
            return []
        
        return await self._retriever.ainvoke(query)

# Create a singleton instance of the service
chroma_service = ChromaService()
