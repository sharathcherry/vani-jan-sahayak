"""
Root alias for deploy.voice_rag_server.
Guarantees compatibility whether Render's start command is:
  - uvicorn main:app
  - uvicorn voice_rag_server:app
  - uvicorn deploy.voice_rag_server:app
"""
import sys
from pathlib import Path

# Ensure deploy directory is on sys.path
_deploy_dir = str(Path(__file__).resolve().parent / "deploy")
if _deploy_dir not in sys.path:
    sys.path.insert(0, _deploy_dir)

from deploy.voice_rag_server import *
from deploy.voice_rag_server import app
