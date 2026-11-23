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
from fastapi import FastAPI, Form, Request, BackgroundTasks, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse, FileResponse
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

# Audio cache directory for outbound voice notes
AUDIO_CACHE_DIR = Path(os.getenv("AUDIO_CACHE_DIR", "/tmp/vani_audio_cache"))
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
# Language Detection & TTS (Text-to-Speech)
# ---------------------------------------------------------------------------

def detect_indic_lang(text: str) -> str:
    """Detect Indian language from unicode script range."""
    for char in text:
        cp = ord(char)
        if 0x0C00 <= cp <= 0x0C7F: return "te"  # Telugu
        if 0x0900 <= cp <= 0x097F: return "hi"  # Hindi / Devanagari
        if 0x0B80 <= cp <= 0x0BFF: return "ta"  # Tamil
        if 0x0C80 <= cp <= 0x0CFF: return "kn"  # Kannada
        if 0x0980 <= cp <= 0x09FF: return "bn"  # Bengali
        if 0x0A80 <= cp <= 0x0AFF: return "gu"  # Gujarati
        if 0x0D00 <= cp <= 0x0D7F: return "ml"  # Malayalam
    return "en"


def synthesize_speech(text: str, lang: str = "en") -> str:
    """
    Synthesize speech audio for the response text.
    Supports Sarvam AI (preferred for Indic), ElevenLabs, and gTTS (default).
    Returns the saved audio filename (e.g. 'vani_12345.mp3') or empty string.
    """
    if not text or not text.strip():
        return ""

    audio_id = f"vani_{uuid.uuid4().hex[:12]}"
    filename = f"{audio_id}.mp3"
    filepath = AUDIO_CACHE_DIR / filename

    # 1. Sarvam AI (India's premier Indic neural voice platform)
    sarvam_key = os.getenv("SARVAM_API_KEY")
    if sarvam_key and lang in ["te", "hi", "ta", "kn", "bn", "mr", "gu", "pa", "ml"]:
        try:
            sarvam_lang_code = f"{lang}-IN"
            headers = {"api-subscription-key": sarvam_key, "Content-Type": "application/json"}
            payload = {
                "inputs": [text[:500]],
                "target_language_code": sarvam_lang_code,
                "speaker": "meera",
                "model": "bulbul:v1"
            }
            r = httpx.post("https://api.sarvam.ai/text-to-speech", json=payload, headers=headers, timeout=12.0)
            if r.status_code == 200:
                import base64
                audio_bytes = base64.b64decode(r.json()["audios"][0])
                filepath.write_bytes(audio_bytes)
                log.info("Synthesized voice note via Sarvam AI (%d bytes): %s", len(audio_bytes), filename)
                return filename
        except Exception as se:
            log.warning("Sarvam TTS failed (%s), falling back to gTTS", se)

    # 2. ElevenLabs Multilingual v2
    eleven_key = os.getenv("ELEVENLABS_API_KEY")
    if eleven_key:
        try:
            voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
            headers = {"xi-api-key": eleven_key, "Content-Type": "application/json"}
            payload = {
                "text": text[:500],
                "model_id": "eleven_multilingual_v2",
            }
            r = httpx.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}", json=payload, headers=headers, timeout=15.0)
            if r.status_code == 200:
                filepath.write_bytes(r.content)
                log.info("Synthesized voice note via ElevenLabs (%d bytes): %s", len(r.content), filename)
                return filename
        except Exception as ee:
            log.warning("ElevenLabs TTS failed (%s), falling back to gTTS", ee)

    # 3. Default built-in gTTS (zero-cost, reliable, supports all Indian languages)
    try:
        from gtts import gTTS
        # Clean markdown asterisks and hash symbols before TTS
        clean_text = text.replace("*", "").replace("#", "").replace("_", "").strip()
        tld = "co.in" if lang == "en" else "com"
        tts = gTTS(text=clean_text[:350], lang=lang, tld=tld, slow=False)
        tts.save(str(filepath))
        log.info("Synthesized voice note via gTTS (%d bytes, lang=%s): %s", filepath.stat().st_size, lang, filename)
        return filename
    except Exception as ge:
        log.error("gTTS speech synthesis failed: %s", ge)
        return ""


# ---------------------------------------------------------------------------
# Core: Answer Generation (Groq / Bedrock)
# ---------------------------------------------------------------------------

def generate_answer(query: str, chunks: list[dict]) -> str:
    """Generate a concise spoken answer in the citizen's native language."""
    context = "\n\n".join([f"Scheme: {c['scheme_id']}\n{c['text']}" for c in chunks])
    lang = detect_indic_lang(query)
    lang_names = {
        "te": "Telugu (తెలుగు)",
        "hi": "Hindi (हिंदी)",
        "ta": "Tamil (தமிழ்)",
        "kn": "Kannada (ಕನ್ನಡ)",
        "bn": "Bengali (বাংলা)",
        "mr": "Marathi (मराठी)",
        "gu": "Gujarati (ગુજરાતી)",
        "ml": "Malayalam (മലയാളം)",
        "en": "Indian English",
    }
    target_lang = lang_names.get(lang, "Indian English")

    prompt = f"""You are Vani (Jan Sahayak), an empathetic voice assistant for Indian Government schemes (MyScheme / Jan Sahayak).
A citizen reached out with the following voice note: "{query}"

Retrieved Government Schemes Knowledge:
{context}

CRITICAL RULES:
1. RESPOND ENTIRELY IN {target_lang}. Use the native script for that language.
2. Address the citizen respectfully by name if provided in their message.
3. Recommend 1 to 3 relevant schemes from the context matching their specific profile (caste, location, occupation, land, education).
4. State the core benefit and key eligibility clearly.
5. Keep the explanation conversational, spoken-friendly, clear, and under 80 words so it speaks naturally in a voice note.
6. Do NOT invent schemes not in the context.

Spoken Response ({target_lang}):"""

    # 1. Primary: Groq LLM (instant ~200-300ms latency)
    if os.getenv("GROQ_API_KEY"):
        try:
            groq_resp = get_groq().chat.completions.create(
                model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=220,
                temperature=0.1,
            )
            answer = groq_resp.choices[0].message.content.strip()
            log.info("Generated answer via Groq (%d chars)", len(answer))
            return answer
        except Exception as ge:
            log.warning("Groq generation failed (%s). Falling back to Bedrock...", ge)

    # 2. Secondary: Bedrock
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
        log.warning("Bedrock generation unavailable (%s). Falling back to scheme summary.", e)

    # 3. Fallback: Extracted scheme text
    if chunks:
        top_text = chunks[0].get("text", "")
        clean_text = " ".join(top_text.split())
        if len(clean_text) > 300:
            clean_text = clean_text[:300] + "..."
        return f"Regarding government schemes: {clean_text}"
    return "I'm sorry, I couldn't find any relevant government scheme for your query. Please call 1800-11-1555."


# ---------------------------------------------------------------------------
# Core: Voice Transcription (Groq Whisper)
# ---------------------------------------------------------------------------

def transcribe_audio_url(audio_url: str, call_sid: str = "") -> str:
    """
    Download audio and transcribe with Groq Whisper.
    Uses whisper-large-v3-turbo with fallback to whisper-large-v3.
    Includes Indic scheme prompt biasing for maximum phonetic accuracy.
    """
    log.info("Downloading audio (SID=%s): %s", call_sid, audio_url)
    try:
        # Only attach Twilio auth if downloading directly from twilio.com
        auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if ("twilio.com" in audio_url and TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN) else None
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = httpx.get(
            audio_url,
            auth=auth,
            headers=headers,
            timeout=25.0,
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

    # Multilingual balanced scheme prompt in Latin transliteration to prevent single-language script biasing
    scheme_prompt = (
        "Government schemes, PM Kisan Samman Nidhi, Rythu Bandhu, Rythu Bharosa, "
        "Pradhan Mantri Awas Yojana, Kisan Credit Card KCC, scholarships, ration card, "
        "subsidy, pension, farmer, agricultural loan, student, weaver, handloom artisan, "
        "caste certificate, backward classes, EWS, Dalit Bandhu, Mudra loan, Jan Sahayak"
    )

    # 1. Primary: whisper-large-v3-turbo
    try:
        log.info(
            "Transcribing audio (%d bytes) with Groq whisper-large-v3-turbo...",
            len(resp.content),
        )
        transcription = groq.audio.transcriptions.create(
            file=file_payload,
            model="whisper-large-v3-turbo",
            prompt=scheme_prompt,
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
            prompt=scheme_prompt,
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
    background_tasks: BackgroundTasks = None,
    RecordingUrl: str = Form(""),
    CallSid: str = Form(""),
    From: str = Form(""),
    To: str = Form(""),
    RecordingDuration: str = Form(""),
    RecordingSid: str = Form(""),
    Digits: str = Form(""),
):
    """
    Step 2: Recording is ready callback from Twilio.
    Form parameters: CallSid, From, To, RecordingUrl, RecordingDuration.
    Transcribe (Groq Whisper) + retrieve (Qdrant) + generate answer (Groq/Bedrock), then respond with TTS TwiML.
    Handles missing or unreachable recording URLs gracefully without crashing.
    """
    log.info(
        "Recording ready callback: CallSid=%s From=%s To=%s Duration=%ss Url=%s",
        CallSid, From, To, RecordingDuration, RecordingUrl
    )

    # 1. Error handling: Missing RecordingUrl
    if not RecordingUrl or not RecordingUrl.strip():
        log.warning("Missing RecordingUrl in webhook callback for CallSid=%s", CallSid)
        response = VoiceResponse()
        response.say(
            "We did not receive any audio recording. Please call back and try again.",
            voice="Polly.Aditi",
            language="en-IN",
        )
        return PlainTextResponse(str(response), media_type="application/xml")

    # 2. Format proper audio URL
    if not RecordingUrl.endswith((".wav", ".mp3", ".ogg")):
        audio_url = f"{RecordingUrl}.wav"
    else:
        audio_url = RecordingUrl

    # 3. Transcribe audio → text
    call_id = CallSid.strip() if CallSid else f"call-{int(time.time())}"
    query_text = transcribe_audio_url(audio_url, call_id)

    if not query_text.strip():
        log.warning("Audio transcription returned empty text or failed for audio_url=%s", audio_url)
        response = VoiceResponse()
        response.say(
            "Sorry, I couldn't understand your question or could not retrieve the audio recording. Please try calling again.",
            voice="Polly.Aditi",
            language="en-IN",
        )
        return PlainTextResponse(str(response), media_type="application/xml")

    log.info("Query: %s", query_text)

    # 4. Retrieve from Qdrant
    try:
        chunks = hybrid_retrieve(query_text)
    except Exception as e:
        log.error("Qdrant retrieval error: %s", e)
        chunks = []

    # 5. Generate answer
    if chunks:
        answer = generate_answer(query_text, chunks)
    else:
        answer = (
            "I'm sorry, I couldn't find any relevant government scheme for your query. "
            "Please try rephrasing or call the helpline at 1800-11-1555."
        )

    # 6. Synthesize native audio for voice playback
    server_base = os.getenv("SERVER_BASE_URL", "https://vani-voice-rag.onrender.com").rstrip("/")
    lang = detect_indic_lang(answer if answer else query_text)
    audio_file = synthesize_speech(answer, lang)

    # 7. Respond with Voice TwiML
    response = VoiceResponse()
    if audio_file:
        audio_url = f"{server_base}/audio/{audio_file}"
        response.play(audio_url)
        log.info("Playing synthesized voice note: %s", audio_url)
    else:
        response.say(answer, voice="Polly.Aditi", language="en-IN")

    log.info("Responded to CallSid=%s with TwiML", CallSid)
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
    Returns both a structured WhatsApp text answer and a voice note media attachment.
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
        answer = "Sorry, no relevant scheme found matching your profile. Please visit the nearest CSC/MeeSeva center or call 1800-11-1555."

    # Synthesize native language voice note audio
    server_base = os.getenv("SERVER_BASE_URL", "https://vani-voice-rag.onrender.com").rstrip("/")
    lang = detect_indic_lang(answer if answer else query_text)
    audio_file = synthesize_speech(answer, lang)

    # Attach both formatted text and audio media voice note to TwiML
    msg = response.message()
    msg.body(f"🏛️ *GovSchemes Answer:*\n\n{answer}")
    if audio_file:
        audio_url = f"{server_base}/audio/{audio_file}"
        msg.media(audio_url)
        log.info("Attached voice note audio to WhatsApp reply: %s", audio_url)

    return PlainTextResponse(str(response), media_type="application/xml")


# ---------------------------------------------------------------------------
# Audio Stream / Serving Endpoint
# ---------------------------------------------------------------------------

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve cached TTS audio voice notes."""
    safe_name = os.path.basename(filename)
    filepath = AUDIO_CACHE_DIR / safe_name
    if filepath.exists():
        return FileResponse(filepath, media_type="audio/mpeg")
    return PlainTextResponse("Audio not found", status_code=404)


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


@app.post("/debug/tts")
async def debug_tts(request: Request):
    """Debug TTS speech synthesis."""
    import traceback
    body = await request.json()
    text = body.get("text", "నమస్కారం, ఇది పరీక్షా సందేశం.")
    lang = body.get("lang", detect_indic_lang(text))
    info = {"text": text, "lang": lang}
    try:
        from gtts import gTTS
        clean_text = text.replace("*", "").replace("#", "").replace("_", "").strip()
        tld = "co.in" if lang == "en" else "com"
        tts = gTTS(text=clean_text[:350], lang=lang, tld=tld, slow=False)
        filepath = AUDIO_CACHE_DIR / "debug_test.mp3"
        tts.save(str(filepath))
        info["saved"] = True
        info["bytes"] = filepath.stat().st_size
    except Exception as e:
        info["saved"] = False
        info["error"] = str(e)
        info["traceback"] = traceback.format_exc()
    return info
async def debug_transcribe(request: Request):
    """Debug audio download and transcription."""
    body = await request.json()
    url = body.get("audio_url", "")
    info = {"url": url}
    try:
        auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if ("twilio.com" in url and TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN) else None
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        resp = httpx.get(url, auth=auth, headers=headers, timeout=20.0, follow_redirects=True)
        info["download_status"] = resp.status_code
        info["download_bytes"] = len(resp.content)
        info["content_type"] = resp.headers.get("content-type")
    except Exception as e:
        info["download_error"] = str(e)
        return info

    try:
        text = transcribe_audio_url(url, "debug_test")
        info["transcription"] = text
    except Exception as e:
        info["transcribe_error"] = str(e)
    return info


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    log.info("Starting Vani Voice RAG server on port %d...", port)
    uvicorn.run("voice_rag_server:app", host="0.0.0.0", port=port, reload=False)
