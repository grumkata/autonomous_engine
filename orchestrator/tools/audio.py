"""
orchestrator/tools/audio.py — Text-to-speech tool.

Providers (in priority order):
  1. ElevenLabs (ELEVENLABS_API_KEY) — best quality, 10K chars/month free
  2. Coqui TTS (local, free) — if installed: pip install TTS
  3. gTTS (Google TTS, free, no key) — fallback, requires internet

All providers save .mp3 to the project workspace.
"""

from __future__ import annotations

import io
from typing import Any

import structlog

from orchestrator.tools.file_store import (
    save_bytes, unique_filename, project_dir, ResourceType,
)

log = structlog.get_logger(__name__)

# Voice name maps
_ELEVENLABS_VOICES = {
    "neutral": "21m00Tcm4TlvDq8ikWAM",     # Rachel
    "female":  "21m00Tcm4TlvDq8ikWAM",     # Rachel
    "male":    "TxGEqnHWrfWFTfGW9XjX",     # Josh
    "british": "29vD33N1BoEC8qNX7WOYG",    # Drew
    "american": "2EiwWnXFnvU5JabPnv8n",    # Clyde
}


async def text_to_speech(
    text: str,
    project_id: str,
    voice: str = "neutral",
    filename: str = "audio",
) -> dict[str, Any]:
    """Convert text to speech and save as MP3."""

    if len(text) > 5000:
        text = text[:5000]
        log.warning("tts.text_truncated", project_id=project_id, original_len=len(text))

    from config import get_settings
    settings = get_settings()

    # ── 1. ElevenLabs ────────────────────────────────────────────────────
    elevenlabs_key = getattr(settings, "elevenlabs_api_key", "")
    if elevenlabs_key:
        try:
            result = await _elevenlabs_tts(text, project_id, voice, filename, elevenlabs_key)
            if result["success"]:
                return result
        except Exception as exc:
            log.warning("tts.elevenlabs_failed", error=str(exc))

    # ── 2. gTTS (Google TTS, free, requires internet) ────────────────────
    try:
        result = await _gtts_tts(text, project_id, voice, filename)
        if result["success"]:
            return result
    except Exception as exc:
        log.warning("tts.gtts_failed", error=str(exc))

    # ── 3. Coqui TTS (local, slow on first run) ──────────────────────────
    try:
        result = await _coqui_tts(text, project_id, voice, filename)
        if result["success"]:
            return result
    except Exception as exc:
        log.warning("tts.coqui_failed", error=str(exc))

    return {
        "tool": "text_to_speech",
        "success": False,
        "error": (
            "No TTS provider available. Options:\n"
            "  1. Add ELEVENLABS_API_KEY to .env (10K chars/month free)\n"
            "  2. Install gTTS: pip install gTTS\n"
            "  3. Install Coqui: pip install TTS"
        ),
        "files_created": [],
    }


async def _elevenlabs_tts(
    text: str, project_id: str, voice: str, filename: str, api_key: str
) -> dict:
    import httpx
    voice_id = _ELEVENLABS_VOICES.get(voice, _ELEVENLABS_VOICES["neutral"])

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            json={
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
        )
        resp.raise_for_status()
        audio_bytes = resp.content

    fname = unique_filename(f"{filename}.mp3", project_id, ResourceType.AUDIO)
    dest  = save_bytes(project_id, fname, audio_bytes, ResourceType.AUDIO)
    rel   = str(dest.relative_to(project_dir(project_id)))

    return {
        "tool": "text_to_speech",
        "success": True,
        "filename": fname,
        "path": rel,
        "provider": "elevenlabs",
        "voice": voice,
        "char_count": len(text),
        "files_created": [rel],
    }


async def _gtts_tts(
    text: str, project_id: str, voice: str, filename: str
) -> dict:
    """Google TTS via gTTS library (free, requires internet)."""
    try:
        from gtts import gTTS
    except ImportError:
        raise RuntimeError("gTTS not installed")

    lang_map = {"british": "en", "american": "en", "neutral": "en",
                "female": "en", "male": "en"}
    tld_map  = {"british": "co.uk", "american": "com"}

    lang = lang_map.get(voice, "en")
    tld  = tld_map.get(voice, "com")

    import asyncio
    loop = asyncio.get_event_loop()

    def _generate() -> bytes:
        buf = io.BytesIO()
        tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
        tts.write_to_fp(buf)
        return buf.getvalue()

    audio_bytes = await loop.run_in_executor(None, _generate)

    fname = unique_filename(f"{filename}.mp3", project_id, ResourceType.AUDIO)
    dest  = save_bytes(project_id, fname, audio_bytes, ResourceType.AUDIO)
    rel   = str(dest.relative_to(project_dir(project_id)))

    return {
        "tool": "text_to_speech",
        "success": True,
        "filename": fname,
        "path": rel,
        "provider": "gtts",
        "voice": voice,
        "char_count": len(text),
        "files_created": [rel],
    }


async def _coqui_tts(
    text: str, project_id: str, voice: str, filename: str
) -> dict:
    """Coqui TTS local model (free, slow first run, no internet needed)."""
    try:
        from TTS.api import TTS as CoquiTTS
    except ImportError:
        raise RuntimeError("Coqui TTS not installed")

    import asyncio
    import tempfile
    from pathlib import Path

    def _generate() -> bytes:
        tts = CoquiTTS("tts_models/en/ljspeech/tacotron2-DDC")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        tts.tts_to_file(text=text, file_path=tmp)
        data = Path(tmp).read_bytes()
        Path(tmp).unlink(missing_ok=True)
        return data

    loop = asyncio.get_event_loop()
    audio_bytes = await loop.run_in_executor(None, _generate)

    fname = unique_filename(f"{filename}.wav", project_id, ResourceType.AUDIO)
    dest  = save_bytes(project_id, fname, audio_bytes, ResourceType.AUDIO)
    rel   = str(dest.relative_to(project_dir(project_id)))

    return {
        "tool": "text_to_speech",
        "success": True,
        "filename": fname,
        "path": rel,
        "provider": "coqui",
        "voice": voice,
        "char_count": len(text),
        "files_created": [rel],
    }
