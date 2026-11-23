"""Speech-to-text through the Deepgram Python SDK (https://github.com/deepgram/deepgram-python-sdk)."""

from __future__ import annotations

from deepgram import AsyncDeepgramClient

from . import config


async def transcribe_audio(audio: bytes, language: str | None = None) -> str:
    """Send recorded audio bytes (webm/opus, mp4, wav...) to Deepgram and return the text."""
    if not config.DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY is not set in .env")
    client = AsyncDeepgramClient(api_key=config.DEEPGRAM_API_KEY)
    options = {"model": "nova-3", "smart_format": True, "punctuate": True}
    if language:
        options["language"] = language
    else:
        options["detect_language"] = True
    response = await client.listen.v1.media.transcribe_file(request=audio, **options)
    channels = response.results.channels
    if not channels or not channels[0].alternatives:
        return ""
    return (channels[0].alternatives[0].transcript or "").strip()


async def live_transcribe(ws, language: str | None = None) -> None:
    """Relay a browser WebSocket to Deepgram live transcription.

    The browser sends binary audio chunks (MediaRecorder webm/opus) and finally the text
    "stop". We reply with JSON: {"text": "...", "final": bool} while the user speaks.
    """
    import asyncio

    from starlette.websockets import WebSocketDisconnect

    if not config.DEEPGRAM_API_KEY:
        await ws.send_json({"error": "DEEPGRAM_API_KEY is not set in .env"})
        return
    client = AsyncDeepgramClient(api_key=config.DEEPGRAM_API_KEY)
    opts = {"model": "nova-3", "interim_results": "true", "smart_format": "true", "punctuate": "true"}
    if language:
        opts["language"] = language
    async with client.listen.v1.connect(**opts) as dg:

        async def downstream() -> None:
            async for msg in dg:
                if getattr(msg, "type", "") != "Results":
                    continue
                alts = msg.channel.alternatives
                text = (alts[0].transcript if alts else "").strip()
                if text:
                    await ws.send_json({"text": text, "final": bool(msg.is_final)})

        reader = asyncio.create_task(downstream())
        try:
            while True:
                data = await ws.receive()
                if data.get("type") == "websocket.disconnect":
                    break
                if data.get("bytes"):
                    await dg.send_media(data["bytes"])
                elif data.get("text") == "stop":
                    break
            await dg.send_finalize()
            await dg.send_close_stream()
            try:
                await asyncio.wait_for(reader, timeout=5)
            except asyncio.TimeoutError:
                reader.cancel()
        except WebSocketDisconnect:
            reader.cancel()
        finally:
            if not reader.done():
                reader.cancel()
