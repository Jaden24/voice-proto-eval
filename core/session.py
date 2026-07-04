"""Runs one complete voice agent interaction: input -> STT -> LLM -> TTS/text.

Usage:
    python core/session.py --text "오늘 너무 힘들어"
    python core/session.py --file path/to/audio.wav
    python core/session.py                          # records from mic
    python core/session.py --no-tts
    python core/session.py --lang ko
"""
import argparse
import json
import os
import sys
import time

import stt
import llm
import tts

SESSION_MINUTES = 10  # for the daily/monthly cost projection shown at the end
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "sessions.jsonl")


def run_session(text: str = None, file_path: str = None, no_tts: bool = False,
                 lang: str = None, mic_seconds: float = 10.0) -> dict:
    stt_result = None

    if text is not None:
        transcript = text
        detected_language = lang or "unknown"
        input_duration = 0.0
        stt_cost = 0.0
    elif file_path is not None:
        stt_result = stt.transcribe_file(file_path, language_hint=lang)
        transcript = stt_result["transcript"]
        detected_language = stt_result["detected_language"]
        input_duration = stt_result["duration_seconds"]
        stt_cost = stt_result["cost_usd"]
    else:
        stt_result = stt.transcribe_mic(seconds=mic_seconds, language_hint=lang)
        transcript = stt_result["transcript"]
        detected_language = stt_result["detected_language"]
        input_duration = stt_result["duration_seconds"]
        stt_cost = stt_result["cost_usd"]

    print(f"\n> {transcript}\n")

    llm_result = llm.get_response(transcript, language_hint=lang or detected_language)

    print(f"\n{llm_result['response_text']}\n")

    tts_result = {"char_count": 0, "cost_usd": 0.0, "voice_id": None, "played": False}
    if not no_tts:
        tts_result = tts.speak(llm_result["response_text"], language=llm_result["language"])

    total_cost = stt_cost + llm_result["cost_usd"] + tts_result["cost_usd"]

    log_session(transcript, detected_language, llm_result, tts_result, total_cost)

    return {
        "input_duration": input_duration,
        "detected_language": detected_language,
        "llm_result": llm_result,
        "stt_cost": stt_cost,
        "llm_cost": llm_result["cost_usd"],
        "tts_cost": tts_result["cost_usd"],
        "total_cost": total_cost,
    }


def log_session(transcript: str, detected_language: str, llm_result: dict, tts_result: dict, total_cost: float):
    """Append this session's transcript + response to logs/sessions.jsonl for later review."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "transcript": transcript,
        "detected_language": detected_language,
        "response_text": llm_result["response_text"],
        "detected_emotion": llm_result["detected_emotion"],
        "suggested_action": llm_result["suggested_action"],
        "risk_flag": llm_result["risk_flag"],
        "tts_played": tts_result["played"],
        "total_cost": total_cost,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def print_summary(result: dict):
    per_second_cost = result["total_cost"] / result["input_duration"] if result["input_duration"] else None
    session_seconds = SESSION_MINUTES * 60

    if per_second_cost is not None:
        projected_session_cost = per_second_cost * session_seconds
    else:
        # text-only session with no audio duration: use the actual total cost as the per-session estimate
        projected_session_cost = result["total_cost"]

    daily_cost = projected_session_cost
    monthly_cost = daily_cost * 30

    bar = "─" * 43
    print(bar)
    print("SESSION SUMMARY")
    print(bar)
    print(f"Input duration     : {result['input_duration']:.1f}s")
    print(f"Detected language  : {result['detected_language']}")
    print(f"Detected emotion   : {result['llm_result']['detected_emotion']}")
    print(f"Risk flag          : {str(result['llm_result']['risk_flag']).lower()}")
    print()
    print(f"STT cost           : ${result['stt_cost']:.4f}")
    print(f"LLM cost           : ${result['llm_cost']:.4f}")
    print(f"TTS cost           : ${result['tts_cost']:.4f}")
    print("─" * 17)
    print(f"Total session cost : ${result['total_cost']:.4f}")
    print()
    print(f"Projected daily cost (10 min/day): ${daily_cost:.2f}")
    print(f"Projected monthly cost (30 days) : ${monthly_cost:.2f}")
    print(bar)


def main():
    parser = argparse.ArgumentParser(description="Run one voice agent session.")
    parser.add_argument("--file", type=str, default=None, help="Path to audio file (.wav/.mp3)")
    parser.add_argument("--text", type=str, default=None, help="Text input, skips STT")
    parser.add_argument("--no-tts", action="store_true", help="Print response text only, no audio")
    parser.add_argument("--lang", type=str, default=None, help="Force language override, e.g. ko")
    parser.add_argument("--mic-seconds", type=float, default=10.0, help="Mic recording duration in seconds")
    args = parser.parse_args()

    if args.file and args.text:
        print("Error: pass either --file or --text, not both.", file=sys.stderr)
        sys.exit(1)

    result = run_session(
        text=args.text,
        file_path=args.file,
        no_tts=args.no_tts,
        lang=args.lang,
        mic_seconds=args.mic_seconds,
    )
    print_summary(result)


if __name__ == "__main__":
    main()
