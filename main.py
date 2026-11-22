"""
Vani (Jan Sahayak) - Cloud Voice RAG Webhook Server
Production entry point for Render.com and cloud containers
"""
import os
import uvicorn
from deploy.voice_rag_server import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
