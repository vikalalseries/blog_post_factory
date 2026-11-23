"""Speech-to-text through Deepgram's plain HTTP API (httpx only, no SDK).

Designed for Vercel: serverless Python functions cannot hold a WebSocket open, so the
server never streams audio. It does two short, stateless HTTP things instead:

  transcribe()   one short recording in -> text out          (POST /api/transcribe)
  voice_token()  a 60 second temporary token for the browser (POST /api/voice-token)
                 so the browser can stream to Deepgram itself WITHOUT ever seeing the real key.

The real key (DEEPGRAM_API_KEY in .env / Vercel environment variables) stays on the server.
"""

from __future__ import annotations

import httpx

from . import config

LISTEN_URL = "https://api.deepgram.com/v1/listen"
GRANT_URL = "https://api.deepgram.com/v1/auth/grant"


class SpeechUnavailable(Exception):
    """The Deepgram key is not configured (HTTP 503)."""


class SpeechFailed(Exception):
    """Deepgram rejected or failed the request (HTTP 502)."""


def _auth() -> dict[str, str]:
    if not config.DEEPGRAM_API_KEY:
        raise SpeechUnavailable("Voice typing is not configured: set DEEPGRAM_API_KEY in .env (or in Vercel's Environment Variables).")
    return {"Authorization": f"Token {config.DEEPGRAM_API_KEY}"}


async def transcribe(audio: bytes, content_type: str = "audio/webm") -> str:
    """Send one short recording to Deepgram and return the transcript (may be empty)."""
    headers = {**_auth(), "Content-Type": content_type or "audio/webm"}
    params = {"model": "nova-3", "language": "multi", "smart_format": "true", "punctuate": "true"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(LISTEN_URL, params=params, headers=headers, content=audio)
        resp.raise_for_status()
        channels = resp.json().get("results", {}).get("channels", [])
        alts = channels[0].get("alternatives", []) if channels else []
        return (alts[0].get("transcript", "") if alts else "").strip()
    except httpx.HTTPStatusError as exc:
        raise SpeechFailed(f"Deepgram returned {exc.response.status_code}.") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise SpeechFailed(f"Could not reach Deepgram: {exc.__class__.__name__}.") from exc


async def voice_token(ttl_seconds: int = 60) -> str:
    """Ask Deepgram for a short-lived token the browser can use for live streaming.

    Needs a key with the Member role or higher; with a weaker key Deepgram answers 403, which
    becomes SpeechFailed and the browser falls back to short recordings via /api/transcribe.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(GRANT_URL, headers=_auth(), json={"ttl_seconds": ttl_seconds})
        resp.raise_for_status()
        token = resp.json().get("access_token")
    except httpx.HTTPStatusError as exc:
        raise SpeechFailed(f"Deepgram refused to issue a token ({exc.response.status_code}).") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise SpeechFailed(f"Could not reach Deepgram: {exc.__class__.__name__}.") from exc
    if not token:
        raise SpeechFailed("Deepgram returned no token.")
    return token
