#!/usr/bin/env python3
"""
Startup script for Bella API with ChromaDB
"""
import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Disable anonymous telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

if __name__ == "__main__":
    print("🚀 Starting Bella API with ChromaDB...")
    print("📊 Vector Database: ChromaDB")
    print("🗂️  Data Directory: ./chroma_db")
    print("📁 Documents Directory: ./company_docs")
    
    port = int(os.getenv("PORT", 8001))
    
    uvicorn.run(
        "trial_chroma:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
