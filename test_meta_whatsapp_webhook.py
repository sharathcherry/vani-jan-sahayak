import os
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add deploy directory to path
deploy_dir = Path(__file__).resolve().parent / "deploy"
sys.path.insert(0, str(deploy_dir))

from voice_rag_server import app, META_VERIFY_TOKEN, transcribe_audio_bytes

client = TestClient(app)

def test_meta_webhook_verification_success():
    """Test Meta webhook verification GET handshake."""
    challenge = "1158201444"
    resp = client.get(
        "/meta/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": META_VERIFY_TOKEN,
            "hub.challenge": challenge,
        }
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert resp.text == challenge, f"Expected challenge {challenge}, got {resp.text}"
    print("[PASS] Meta Webhook Verification Success (HTTP 200)")

def test_meta_webhook_verification_failure():
    """Test Meta webhook verification failure on invalid token."""
    resp = client.get(
        "/meta/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong_token",
            "hub.challenge": "12345",
        }
    )
    assert resp.status_code == 403, f"Expected 403, got {resp.status_code}"
    print("[PASS] Meta Webhook Verification Rejection on Bad Token (HTTP 403)")

def test_meta_webhook_incoming_text_payload():
    """Test Meta webhook incoming message POST payload."""
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "1234567890",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15551234567",
                                "phone_number_id": "987654321",
                            },
                            "contacts": [{"profile": {"name": "Ramesh"}, "wa_id": "919876500002"}],
                            "messages": [
                                {
                                    "from": "919876500002",
                                    "id": "wamid.HBgLMTIzNDU2",
                                    "timestamp": "1732356000",
                                    "type": "text",
                                    "text": {"body": "Kisan Credit Card scheme details"},
                                }
                            ],
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }

    resp = client.post("/meta/webhook", json=payload)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert data.get("status") == "ok", f"Expected ok status, got {data}"
    print("[PASS] Meta Incoming Text Webhook Handled Immediately (HTTP 200)")

def test_meta_webhook_incoming_audio_payload():
    """Test Meta webhook incoming audio voice note payload."""
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "1234567890",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15551234567",
                                "phone_number_id": "987654321",
                            },
                            "contacts": [{"profile": {"name": "Murugan"}, "wa_id": "919876500004"}],
                            "messages": [
                                {
                                    "from": "919876500004",
                                    "id": "wamid.HBgLMjM0NTY3",
                                    "timestamp": "1732356005",
                                    "type": "audio",
                                    "audio": {
                                        "id": "media_audio_test_123",
                                        "mime_type": "audio/ogg; codecs=opus",
                                    },
                                }
                            ],
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }

    resp = client.post("/meta/webhook", json=payload)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert data.get("status") == "ok", f"Expected ok status, got {data}"
    print("[PASS] Meta Incoming Voice Note Webhook Handled Immediately (HTTP 200)")

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 RUNNING META WHATSAPP CLOUD API INTEGRATION TESTS")
    print("=" * 60)
    test_meta_webhook_verification_success()
    test_meta_webhook_verification_failure()
    test_meta_webhook_incoming_text_payload()
    test_meta_webhook_incoming_audio_payload()
    print("=" * 60)
    print("🎉 ALL META CLOUD API TESTS PASSED WITH 0 ERRORS!")
    print("=" * 60)
