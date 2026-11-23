"""
voice_rag_server.py
--------------------
Twilio Voice / WhatsApp → Groq Whisper → Qdrant Hybrid RAG → Bedrock → Twilio TTS

Flow:
  1. User calls Twilio number or sends WhatsApp voice note
  2. Twilio records voice message and hits /voice/incoming or /whatsapp/incoming
  3. Recording webhook hits /voice/recording_ready or /whatsapp/incoming with audio URL
  4. Groq Whisper (whisper-large-v3-turbo) converts audio → text
  5. Qdrant hybrid retrieval finds relevant scheme chunks
  6. Bedrock Nova generates a concise answer
  7. Server returns TwiML with <Say> (TTS) or WhatsApp message response

Run:
  uvicorn voice_rag_server:app --host 0.0.0.0 --port 8000

Expose via ngrok for local dev:
  ngrok http 8000
  Set Twilio webhook → https://<ngrok-url>/voice/incoming
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path

import boto3
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request, BackgroundTasks
from fastapi.responses import PlainTextResponse, JSONResponse
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import FusionQuery, Fusion, Prefetch, SparseVector
from fastembed import TextEmbedding, SparseTextEmbedding
from twilio.twiml.voice_response import VoiceResponse, Record, Gather
from twilio.rest import Client as TwilioClient

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="GovSchemes Voice RAG", version="1.0.0")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QDRANT_LOCAL_PATH = os.getenv("QDRANT_PATH", "qdrant_hybrid_db")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "schemes_small")
DENSE_MODEL = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "20"))
TOP_K = int(os.getenv("TOP_K", "5"))

AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "eu-north-1"))
BEDROCK_LLM_MODEL = os.getenv("BEDROCK_LLM_MODEL", "eu.amazon.nova-lite-v1:0")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# S3 bucket for Transcribe (needs a bucket in same region)
TRANSCRIBE_S3_BUCKET = os.getenv("BEDROCK_S3_BUCKET", os.getenv("TRANSCRIBE_S3_BUCKET", "government-scheme"))

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------

_qdrant_client: QdrantClient | None = None
_dense_emb: TextEmbedding | None = None
_sparse_emb: SparseTextEmbedding | None = None
_bedrock_client = None
_transcribe_client = None
_s3_client = None
_twilio_client: TwilioClient | None = None
_groq_client: Groq | None = None


def get_qdrant():
    global _qdrant_client
    if _qdrant_client is None:
        cloud_url = os.getenv("QDRANT_CLOUD_URL")
        cloud_key = os.getenv("QDRANT_CLOUD_API_KEY")
        if cloud_url and cloud_key:
            log.info("Connecting to Qdrant Cloud: %s", cloud_url)
            _qdrant_client = QdrantClient(url=cloud_url, api_key=cloud_key, timeout=30)
        else:
            log.info("Using local Qdrant at: %s", QDRANT_LOCAL_PATH)
            _qdrant_client = QdrantClient(path=QDRANT_LOCAL_PATH)
    return _qdrant_client


def get_dense_embedder():
    global _dense_emb
    if _dense_emb is None:
        log.info("Loading dense embedder: %s", DENSE_MODEL)
        # Auto-detect GPU
        try:
            import onnxruntime as ort
            providers = (
                ["CUDAExecutionProvider"]
                if "CUDAExecutionProvider" in ort.get_available_providers()
                else ["CPUExecutionProvider"]
            )
        except ImportError:
            providers = ["CPUExecutionProvider"]
        _dense_emb = TextEmbedding(model_name=DENSE_MODEL, providers=providers)
        log.info("Dense embedder loaded (providers: %s)", providers)
    return _dense_emb


def get_sparse_embedder():
    global _sparse_emb
    if _sparse_emb is None:
        _sparse_emb = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_emb


def get_bedrock():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _bedrock_client


def get_transcribe():
    global _transcribe_client
    if _transcribe_client is None:
        _transcribe_client = boto3.client("transcribe", region_name=AWS_REGION)
    return _transcribe_client


def get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client


def get_twilio():
    global _twilio_client
    if _twilio_client is None:
        _twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    return _twilio_client


def get_groq():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


# ---------------------------------------------------------------------------
# Warm up models on startup (prevents first-call hang)
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def warmup():
    log.info("Warming up models...")
    try:
        get_dense_embedder()
        get_sparse_embedder()
        get_qdrant()
        log.info("Models and vector database ready.")
    except Exception as e:
        log.warning("Warmup initial connection notice (will load on first request): %s", e)


# ---------------------------------------------------------------------------
# Core: Encoding
# ---------------------------------------------------------------------------

def encode_dense(text: str) -> list[float]:
    return list(get_dense_embedder().embed([text]))[0].tolist()


def encode_sparse(text: str) -> SparseVector:
    sp = list(get_sparse_embedder().embed([text]))[0]
    return SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())


# ---------------------------------------------------------------------------
# Core: Hybrid Retrieval
# ---------------------------------------------------------------------------

def hybrid_retrieve(query: str) -> list[dict]:
    """Hybrid dense+sparse retrieval with RRF fusion."""
    dense_vec = encode_dense(query)
    sparse_vec = encode_sparse(query)

    results = get_qdrant().query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=CANDIDATE_K),
            Prefetch(query=sparse_vec, using="sparse", limit=CANDIDATE_K),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=TOP_K,
        with_payload=True,
    )

    chunks = []
    for point in results.points:
        chunks.append({
            "text": point.payload.get("text", ""),
            "scheme_id": point.payload.get("scheme_id", ""),
            "score": point.score or 0.0,
        })

    log.info("Retrieved %d chunks for query: %s...", len(chunks), query[:60])
    return chunks


# ---------------------------------------------------------------------------
# Core: Answer Generation (Bedrock)
# ---------------------------------------------------------------------------

def generate_answer(query: str, chunks: list[dict]) -> str:
    """Generate a concise spoken answer using Bedrock Nova."""
    context = "\n\n".join(
        [f"Scheme: {c['scheme_id']}\n{c['text']}" for c in chunks]
    )

    prompt = f"""You are a helpful assistant for Indian government schemes. 
A citizen has asked a question via voice call. Answer in simple, clear language 
suitable for text-to-speech. Keep it under 100 words. If you don't know, say so.

Context:
{context}

Question: {query}

Answer (keep it short and spoken-friendly):"""

    try:
        response = get_bedrock().converse(
            modelId=BEDROCK_LLM_MODEL,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 256},
        )
        answer = response["output"]["message"]["content"][0]["text"]
        log.info("Generated answer via Bedrock (%d chars)", len(answer))
        return answer
    except Exception as e:
        log.warning("Bedrock generation unavailable (%s). Falling back to retrieved scheme context.", e)
        if chunks:
            top_text = chunks[0].get("text", "")
            clean_text = " ".join(top_text.split())
            if len(clean_text) > 300:
                clean_text = clean_text[:300] + "..."
            return f"Regarding your query on government schemes: {clean_text}"
        return (
            "I'm sorry, I couldn't find any relevant government scheme for your query. "
            "Please try rephrasing or call the helpline at 1800-11-1555."
        )


# ---------------------------------------------------------------------------
# Core: Voice Transcription (Groq Whisper)
# ---------------------------------------------------------------------------

def transcribe_audio_url(audio_url: str, call_sid: str = "") -> str:
    """
    Download audio from Twilio URL and transcribe with Groq Whisper.
    Uses whisper-large-v3-turbo with fallback to whisper-large-v3.
    Returns transcribed text.
    """
    log.info("Downloading audio from Twilio (SID=%s): %s", call_sid, audio_url)
    try:
        auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN) else None
        resp = httpx.get(
            audio_url,
            auth=auth,
            timeout=30.0,
            follow_redirects=True,
        )
        resp.raise_for_status()
    except Exception as e:
        log.error("Failed to download audio from %s: %s", audio_url, e)
        return ""

    if not resp.content:
        log.warning("Downloaded audio is empty from %s", audio_url)
        return ""

    # Detect content type / format (e.g. WhatsApp ogg vs Twilio Voice wav)
    content_type = resp.headers.get("content-type", "").lower()
    if "ogg" in content_type:
        file_payload = ("audio.ogg", resp.content, "audio/ogg")
    elif "mp3" in content_type or "mpeg" in content_type:
        file_payload = ("audio.mp3", resp.content, "audio/mpeg")
    else:
        file_payload = ("audio.wav", resp.content, "audio/wav")

    groq = get_groq()

    # 1. Primary: whisper-large-v3-turbo
    try:
        log.info(
            "Transcribing audio (%d bytes) with Groq whisper-large-v3-turbo...",
            len(resp.content),
        )
        transcription = groq.audio.transcriptions.create(
            file=file_payload,
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
        )
        text = transcription.text.strip()
        log.info("Transcription result (whisper-large-v3-turbo): %s", text)
        return text
    except Exception as e:
        log.warning(
            "Groq whisper-large-v3-turbo failed (%s), falling back to whisper-large-v3...",
            e,
        )

    # 2. Fallback: whisper-large-v3
    try:
        log.info("Transcribing audio with fallback model whisper-large-v3...")
        transcription = groq.audio.transcriptions.create(
            file=file_payload,
            model="whisper-large-v3",
            response_format="verbose_json",
        )
        text = transcription.text.strip()
        log.info("Transcription result (whisper-large-v3 fallback): %s", text)
        return text
    except Exception as e:
        log.error(
            "Groq Whisper transcription failed completely for SID=%s: %s",
            call_sid,
            e,
        )
        return ""


# ---------------------------------------------------------------------------
# Twilio Webhook Endpoints
# ---------------------------------------------------------------------------

@app.post("/voice/incoming", response_class=PlainTextResponse)
async def voice_incoming(request: Request):
    """
    Step 1: User calls → Twilio hits this endpoint.
    We greet the user and start recording their question.
    """
    response = VoiceResponse()
    response.say(
        "Welcome to the Government Schemes helpline. "
        "Please speak your question after the beep, then press any key or stay silent for 3 seconds.",
        voice="Polly.Aditi",  # Indian English voice
        language="en-IN",
    )
    # Record up to 30 seconds; Twilio will POST to /voice/recording_ready
    response.record(
        max_length=30,
        timeout=3,
        transcribe=False,  # We use AWS Transcribe instead
        action="/voice/recording_ready",
        method="POST",
        play_beep=True,
    )

    log.info("Incoming call — greeting sent")
    return PlainTextResponse(str(response), media_type="application/xml")


@app.post("/voice/recording_ready", response_class=PlainTextResponse)
async def voice_recording_ready(
    request: Request,
    background_tasks: BackgroundTasks,
    RecordingUrl: str = Form(...),
    CallSid: str = Form(...),
    RecordingDuration: str = Form(""),
):
    """
    Step 2: Recording is ready.
    Transcribe + retrieve + generate answer, then respond with TTS.
    """
    log.info("Recording ready: CallSid=%s Duration=%ss", CallSid, RecordingDuration)
    audio_url = f"{RecordingUrl}.wav"

    # Transcribe audio → text
    query_text = transcribe_audio_url(audio_url, CallSid)

    if not query_text.strip():
        response = VoiceResponse()
        response.say(
            "Sorry, I couldn't understand your question. Please try calling again.",
            voice="Polly.Aditi",
            language="en-IN",
        )
        return PlainTextResponse(str(response), media_type="application/xml")

    log.info("Query: %s", query_text)

    # Retrieve from Qdrant
    chunks = hybrid_retrieve(query_text)

    # Generate answer
    if chunks:
        answer = generate_answer(query_text, chunks)
    else:
        answer = (
            "I'm sorry, I couldn't find any relevant government scheme for your query. "
            "Please try rephrasing or call the helpline at 1800-11-1555."
        )

    # Respond with TTS
    response = VoiceResponse()
    response.say(answer, voice="Polly.Aditi", language="en-IN")
    response.say(
        "Thank you for using the Government Schemes helpline. Goodbye!",
        voice="Polly.Aditi",
        language="en-IN",
    )

    log.info("Responded to CallSid=%s", CallSid)
    return PlainTextResponse(str(response), media_type="application/xml")


# ---------------------------------------------------------------------------
# WhatsApp Voice Note Endpoint (Twilio WhatsApp)
# ---------------------------------------------------------------------------

@app.post("/whatsapp/incoming", response_class=PlainTextResponse)
async def whatsapp_incoming(
    request: Request,
    Body: str = Form(""),
    MediaUrl0: str = Form(""),
    MediaContentType0: str = Form(""),
    From: str = Form(""),
    To: str = Form(""),
    MessageSid: str = Form(""),
):
    """
    Handles WhatsApp messages (text or voice notes).
    Twilio webhook for WhatsApp incoming messages.
    """
    from twilio.twiml.messaging_response import MessagingResponse

    log.info("WhatsApp from=%s type=%s MediaUrl0=%s", From, MediaContentType0, MediaUrl0)

    query_text = ""

    # Voice note / media attachment: if MediaUrl0 is provided, transcribe via Groq Whisper
    if MediaUrl0 and MediaUrl0.strip():
        log.info("Transcribing WhatsApp media note from %s (SID=%s): %s", From, MessageSid, MediaUrl0)
        query_text = transcribe_audio_url(MediaUrl0.strip(), MessageSid)

    # Fallback to text message Body if no media attachment or empty transcription
    if not query_text and Body.strip():
        query_text = Body.strip()
        log.info("WhatsApp text query: %s", query_text)

    response = MessagingResponse()

    if not query_text:
        response.message("Sorry, I couldn't understand your message. Please send a text or voice note.")
        return PlainTextResponse(str(response), media_type="application/xml")

    # Retrieve + generate
    chunks = hybrid_retrieve(query_text)
    if chunks:
        answer = generate_answer(query_text, chunks)
    else:
        answer = "Sorry, no relevant scheme found. Try rephrasing your question."

    response.message(f"🏛️ *GovSchemes Answer:*\n\n{answer}")
    return PlainTextResponse(str(response), media_type="application/xml")


# ---------------------------------------------------------------------------
# Debug / Health endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    cloud_url = os.getenv("QDRANT_CLOUD_URL")
    return {
        "service": "Vani Jan Sahayak - Voice RAG Helpline",
        "status": "online",
        "version": "1.0.0",
        "vector_backend": "qdrant_cloud" if cloud_url else "qdrant_local",
        "collection": COLLECTION_NAME,
        "endpoints": {
            "health": "/health",
            "voice_incoming": "/voice/incoming",
            "voice_recording_ready": "/voice/recording_ready",
            "whatsapp_incoming": "/whatsapp/incoming",
            "debug_query": "/debug/query",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    cloud_url = os.getenv("QDRANT_CLOUD_URL")
    return {
        "status": "ok",
        "service": "vani-jan-sahayak",
        "collection": COLLECTION_NAME,
        "backend": "qdrant_cloud" if cloud_url else "qdrant_local",
        "region": AWS_REGION,
        "model": BEDROCK_LLM_MODEL
    }


@app.post("/debug/query")
async def debug_query(request: Request):
    """Test retrieval directly with a text query (no voice)."""
    body = await request.json()
    query = body.get("query", "")
    if not query:
        return JSONResponse({"error": "query required"}, status_code=400)

    chunks = hybrid_retrieve(query)
    answer = generate_answer(query, chunks) if chunks else "No results found."
    return {
        "query": query,
        "answer": answer,
        "chunks_count": len(chunks),
        "top_chunk": chunks[0] if chunks else None,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    log.info("Starting Vani Voice RAG server on port %d...", port)
    uvicorn.run("voice_rag_server:app", host="0.0.0.0", port=port, reload=False)
