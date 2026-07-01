"""TTS wrapper around ElevenLabs Flash v2.5, with graceful no-key fallback."""
import os

from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "eleven_flash_v2_5"
COST_PER_CHAR = 0.00005

DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # ElevenLabs default "Rachel" voice


def speak(text: str, language: str = "en") -> dict:
    """Convert text to speech and play it back immediately.

    If ELEVENLABS_API_KEY is missing, falls back to printing the text.
    """
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID") or DEFAULT_VOICE_ID
    char_count = len(text)
    cost = round(char_count * COST_PER_CHAR, 6)

    if not api_key:
        print(f"\n[TTS skipped - no API key]\n{text}\n")
        return {"char_count": char_count, "cost_usd": cost, "voice_id": voice_id, "played": False}

    import numpy as np
    import sounddevice as sd
    from elevenlabs.client import ElevenLabs

    samplerate = 16000
    client = ElevenLabs(api_key=api_key)
    # Request raw PCM so we can play it back directly via sounddevice, no decoder needed.
    pcm_stream = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=MODEL_ID,
        output_format="pcm_16000",
    )
    pcm_bytes = b"".join(pcm_stream)
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    sd.play(audio, samplerate=samplerate)
    sd.wait()

    return {"char_count": char_count, "cost_usd": cost, "voice_id": voice_id, "played": True}
