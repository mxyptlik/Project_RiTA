#!/usr/bin/env python3
"""
Test script for ChromaDB integration
"""
import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# Disable anonymous telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

DOCS_DIR = "./company_docs"
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-3-large"

async def test_chromadb():
    """Test ChromaDB functionality"""
    print("🧪 Testing ChromaDB integration...")
    
    try:
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        print("✅ OpenAI embeddings initialized")
        
        # Initialize ChromaDB
        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name="bella_documents"
        )
        print("✅ ChromaDB initialized")
        
        # Check existing documents
        collection = vector_store._collection
        count = collection.count()
        print(f"📊 Current document count: {count}")
        
        if count == 0:
            print("📁 Loading documents from", DOCS_DIR)
            
            # Check if docs directory exists
            if not os.path.exists(DOCS_DIR):
                print(f"⚠️  {DOCS_DIR} not found")
                return
            
            # Load documents
            loader = DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
            documents = loader.load()
            print(f"📄 Found {len(documents)} PDF documents")
            
            if documents:
                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
                chunks = text_splitter.split_documents(documents)
                print(f"✂️  Created {len(chunks)} chunks")
                
                # Add to ChromaDB
                vector_store.add_documents(chunks)
                print(f"💾 Added {len(chunks)} documents to ChromaDB")
        
        # Test retrieval
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        test_query = "What services does Heckerbella offer?"
        
        print(f"🔍 Testing query: '{test_query}'")
        results = retriever.invoke(test_query)
        print(f"📋 Retrieved {len(results)} documents")
        
        if results:
            print("\n🎯 Sample result:")
            print(f"Content preview: {results[0].page_content[:200]}...")
            print(f"Metadata: {results[0].metadata}")
        
        print("\n✅ ChromaDB test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_chromadb())
