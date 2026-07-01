"""STT wrapper around OpenAI Whisper (whisper-1)."""
import os
import time
import wave
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

WHISPER_COST_PER_MINUTE = 0.006

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


def _wav_duration_seconds(file_path: str) -> float:
    """Local duration lookup for wav files, used only as a fallback."""
    try:
        with wave.open(file_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        return 0.0


def transcribe_file(file_path: str, language_hint: str = None) -> dict:
    """Transcribe an audio file on disk using whisper-1."""
    client = _get_client()

    with open(file_path, "rb") as audio_file:
        kwargs = {"model": "whisper-1", "file": audio_file, "response_format": "verbose_json"}
        if language_hint:
            kwargs["language"] = language_hint
        result = client.audio.transcriptions.create(**kwargs)

    transcript = result.text
    detected_language = getattr(result, "language", None) or language_hint or "unknown"
    # whisper-1 verbose_json includes duration; fall back to local wav parsing if absent.
    duration = getattr(result, "duration", None)
    if not duration:
        duration = _wav_duration_seconds(file_path) if file_path.lower().endswith(".wav") else 0.0

    cost = round((duration / 60.0) * WHISPER_COST_PER_MINUTE, 6)

    return {
        "transcript": transcript,
        "detected_language": detected_language,
        "duration_seconds": round(duration, 2),
        "cost_usd": cost,
    }


def record_from_mic(seconds: float = 10.0, samplerate: int = 16000) -> str:
    """Record `seconds` of audio from the default mic and save to a temp wav file."""
    import sounddevice as sd
    from scipy.io.wavfile import write as wav_write

    print(f"Recording for {seconds:.1f}s... speak now.")
    recording = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()
    print("Recording finished.")

    out_path = str(Path.cwd() / f"_mic_recording_{int(time.time())}.wav")
    wav_write(out_path, samplerate, recording)
    return out_path


def transcribe_mic(seconds: float = 10.0, language_hint: str = None) -> dict:
    """Record from mic then transcribe. Cleans up the temp file afterward."""
    file_path = record_from_mic(seconds=seconds)
    try:
        return transcribe_file(file_path, language_hint=language_hint)
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass
