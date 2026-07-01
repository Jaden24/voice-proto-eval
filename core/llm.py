"""LLM wrapper around OpenAI GPT-4o with structured JSON output."""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o"
INPUT_COST_PER_TOKEN = 0.000005
OUTPUT_COST_PER_TOKEN = 0.000015

SYSTEM_PROMPT = """You are a warm, empathetic counselor.

Rules you always follow:
- Always validate the user's feelings before offering anything else. Never jump straight to advice.
- Respond in the SAME language the user spoke in. If they wrote in Korean, respond in Korean. If English, respond in English. If mixed, respond primarily in the dominant language.
- Korean tone: soft, conversational 존댓말, like a caring friend. Not formal or clinical.
- English tone: warm, encouraging, casual and human — not corporate, not therapist-textbook.
- End every response with exactly one small, concrete suggested action the person could take. Keep it tiny and doable, not a big life change.
- Never use guilt, blame, or pressure language ("you should have", "you need to", "why didn't you"). No pressure, ever.
- Crisis detection: if the input contains crisis-adjacent language (e.g. "죽고 싶다", "want to disappear", "사라지고 싶다", "don't want to exist", or similar expressions of self-harm/suicidal ideation), you MUST set risk_flag to true and include a crisis helpline in the response_text regardless of anything else. For Korean, mention 자살예방상담전화 1393. For English, mention the 988 Suicide & Crisis Lifeline (call or text 988, US). This rule overrides all other stylistic preferences.

You must respond with ONLY a JSON object, no preamble, no markdown fences, matching exactly this shape:
{
  "response_text": "the actual response to speak/display",
  "detected_emotion": "e.g. burnout, anxiety, loneliness",
  "suggested_action": "the small concrete action mentioned at the end",
  "risk_flag": true or false,
  "language": "language code of the response, e.g. ko or en"
}
"""

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


def get_response(user_text: str, language_hint: str = None) -> dict:
    """Call GPT-4o with the counselor system prompt and return structured output."""
    client = _get_client()

    user_content = user_text
    if language_hint:
        user_content = f"[User's spoken language detected as: {language_hint}]\n{user_text}"

    completion = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    raw = completion.choices[0].message.content
    parsed = json.loads(raw)

    usage = completion.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    cost = round(input_tokens * INPUT_COST_PER_TOKEN + output_tokens * OUTPUT_COST_PER_TOKEN, 6)

    return {
        "response_text": parsed.get("response_text", ""),
        "detected_emotion": parsed.get("detected_emotion", "unknown"),
        "suggested_action": parsed.get("suggested_action", ""),
        "risk_flag": bool(parsed.get("risk_flag", False)),
        "language": parsed.get("language", language_hint or "unknown"),
        "tokens": {"input": input_tokens, "output": output_tokens},
        "cost_usd": cost,
    }
