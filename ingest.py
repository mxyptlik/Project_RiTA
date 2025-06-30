
import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DOCS_DIR = os.getenv("DOCS_DIR", "company_docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "RITA_document"

def ingest_documents():
    """
    Loads documents from the specified directory, splits them into chunks,
    generates embeddings, and ingests them into a persistent ChromaDB collection.
    """
    logger.info("--- Starting document ingestion process ---")

    # 1. Check if documents directory exists
    if not os.path.exists(DOCS_DIR):
        logger.error(f"Error: Document directory not found at '{DOCS_DIR}'.")
        logger.info("Please create the directory and add your PDF files.")
        return

    # 2. Load documents
    logger.info(f"Loading documents from '{DOCS_DIR}'...")
    try:
        loader = DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=True)
        documents = loader.load()
        if not documents:
            logger.warning("No PDF documents found to ingest. The process will stop.")
            return
        logger.info(f"Successfully loaded {len(documents)} document(s).")
    except Exception as e:
        logger.error(f"Failed to load documents: {e}", exc_info=True)
        return

    # 3. Filter out invalid or empty documents
    valid_documents = [doc for doc in documents if hasattr(doc, 'page_content') and doc.page_content.strip()]
    if not valid_documents:
        logger.warning("No valid content found in the loaded documents. Ingestion stopped.")
        return
    logger.info(f"Found {len(valid_documents)} valid documents for processing.")

    # 4. Split documents into chunks
    logger.info("Splitting documents into manageable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
    chunks = text_splitter.split_documents(valid_documents)
    if not chunks:
        logger.warning("Could not create any chunks from the documents. Ingestion stopped.")
        return
    logger.info(f"Created {len(chunks)} chunks from the documents.")

    # 5. Initialize embeddings model
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL}")
    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI Embeddings: {e}", exc_info=True)
        return

    # 6. Initialize ChromaDB and ingest documents
    logger.info(f"Initializing ChromaDB with persistence at: {CHROMA_DB_PATH}")
    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DB_PATH
        )
        logger.info(f"Successfully ingested {len(chunks)} chunks into ChromaDB collection '{COLLECTION_NAME}'.")
    except Exception as e:
        logger.error(f"Failed to ingest documents into ChromaDB: {e}", exc_info=True)
        return

    logger.info("--- Document ingestion process completed successfully! ---")

if __name__ == "__main__":
    # Set the ANONYMIZED_TELEMETRY to False to disable anonymous telemetry
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    ingest_documents()
