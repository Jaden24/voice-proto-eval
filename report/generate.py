"""Reads the latest eval/results/*.json and generates a cost + quality report.

Usage:
    python report/generate.py
"""
import glob
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "eval", "results")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "report", "latest_report.md")

STT_COST_PER_MINUTE = 0.006
TTS_COST_PER_CHAR = 0.00005
MONTHLY_TARGET_PER_USER = 0.50

SESSION_MINUTES = 10.0
# Assumption: an average conversational turn (one user utterance) runs ~30s of
# speech, so a 10-minute session is made up of roughly this many back-and-forth turns.
TURN_SECONDS = 30.0
TURNS_PER_SESSION = SESSION_MINUTES * 60 / TURN_SECONDS

DIMENSION_KEYS = ["empathy_first", "action_included", "no_guilt", "language_match"]
DIMENSION_LABELS = {
    "empathy_first": "empathy_first",
    "action_included": "action_included",
    "no_guilt": "no_guilt",
    "language_match": "language_match",
}


def find_latest_results():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
    if not files:
        raise SystemExit(f"No eval results found in {RESULTS_DIR}. Run `python eval/run_eval.py` first.")
    return files[-1]


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_cost_model(results):
    llm_costs = [r["llm_result"]["cost_usd"] for r in results]
    response_lengths = [len(r["llm_result"]["response_text"]) for r in results]

    avg_llm_cost_per_turn = sum(llm_costs) / len(llm_costs)
    avg_response_chars = sum(response_lengths) / len(response_lengths)
    avg_tts_cost_per_turn = avg_response_chars * TTS_COST_PER_CHAR

    stt_per_session = SESSION_MINUTES * STT_COST_PER_MINUTE
    llm_per_session = avg_llm_cost_per_turn * TURNS_PER_SESSION
    tts_per_session = avg_tts_cost_per_turn * TURNS_PER_SESSION

    stt_per_month = stt_per_session * 30
    llm_per_month = llm_per_session * 30
    tts_per_month = tts_per_session * 30

    total_per_month = stt_per_month + llm_per_month + tts_per_month

    return {
        "avg_llm_cost_per_turn": avg_llm_cost_per_turn,
        "avg_response_chars": avg_response_chars,
        "avg_tts_cost_per_turn": avg_tts_cost_per_turn,
        "stt_per_session": stt_per_session,
        "llm_per_session": llm_per_session,
        "tts_per_session": tts_per_session,
        "stt_per_month": stt_per_month,
        "llm_per_month": llm_per_month,
        "tts_per_month": tts_per_month,
        "total_per_month": total_per_month,
    }


def compute_quality_summary(payload, results):
    summary = payload["summary"]
    total = summary["total_scenarios"]

    dimension_fail_counts = {k: total - summary["by_dimension"][k] for k in DIMENSION_KEYS}
    worst_dimension = max(dimension_fail_counts, key=dimension_fail_counts.get)

    group_scores = {}
    for r in results:
        key = (r["language"], r["persona"])
        j = r["judge_result"]
        group_scores.setdefault(key, []).append(j["total"] / j["max"] if j["max"] else 1.0)

    group_avgs = {k: sum(v) / len(v) for k, v in group_scores.items()}
    worst_group = min(group_avgs, key=group_avgs.get)

    return {
        "summary": summary,
        "worst_dimension": worst_dimension,
        "worst_dimension_fail_count": dimension_fail_counts[worst_dimension],
        "worst_group": worst_group,
        "worst_group_avg": group_avgs[worst_group],
    }


def lang_dimension_avg(results, language, dimension):
    scores = [r["judge_result"]["scores"][dimension] for r in results if r["language"] == language]
    if not scores:
        return None
    return sum(scores) / len(scores)


def generate_observations(cost_model, quality, results):
    observations = []

    ko_empathy = lang_dimension_avg(results, "ko", "empathy_first")
    en_empathy = lang_dimension_avg(results, "en", "empathy_first")
    if ko_empathy is not None and en_empathy is not None and ko_empathy < en_empathy:
        observations.append(
            f"Korean responses scored lower on empathy_first ({ko_empathy:.0%}) than English "
            f"({en_empathy:.0%}) — consider tuning the system prompt for Korean tone."
        )
    elif ko_empathy is not None and en_empathy is not None and en_empathy < ko_empathy:
        observations.append(
            f"English responses scored lower on empathy_first ({en_empathy:.0%}) than Korean "
            f"({ko_empathy:.0%}) — consider tuning the system prompt for English tone."
        )

    risk_summary = quality["summary"]
    if risk_summary["risk_gate_pass"]:
        observations.append(
            f"Risk gate: all {risk_summary['risk_total']} crisis cases handled correctly."
        )
    else:
        observations.append(
            f"RISK GATE FAILURE: only {risk_summary['risk_correct']}/{risk_summary['risk_total']} "
            "crisis cases triggered risk_flag correctly — do not ship this prompt until fixed."
        )

    monthly_per_user = cost_model["total_per_month"]
    if monthly_per_user <= MONTHLY_TARGET_PER_USER:
        observations.append(
            f"Estimated monthly cost per user (${monthly_per_user:.2f}) is within the "
            f"${MONTHLY_TARGET_PER_USER:.2f} target."
        )
    else:
        observations.append(
            f"Estimated monthly cost per user (${monthly_per_user:.2f}) EXCEEDS the "
            f"${MONTHLY_TARGET_PER_USER:.2f} target — consider prompt length optimization or a "
            "cheaper TTS/LLM tier."
        )

    observations.append(
        f"Weakest quality dimension: {quality['worst_dimension']} failed on "
        f"{quality['worst_dimension_fail_count']}/{quality['summary']['total_scenarios']} scenarios."
    )

    lang, persona = quality["worst_group"]
    observations.append(
        f"Lowest-scoring segment: {lang}/{persona} averaged {quality['worst_group_avg']:.0%} "
        "across all dimensions — worth a closer look before broader rollout."
    )

    return observations


def render_cost_table(cost_model):
    lines = []
    lines.append("─" * 55)
    lines.append(f"COST MODEL: {SESSION_MINUTES:.0f} min/day per user "
                  f"(~{TURNS_PER_SESSION:.0f} turns/session assumed)")
    lines.append("─" * 55)
    lines.append(f"{'':<22}{'Per session':<14}{'Per day':<10}{'Per month'}")
    lines.append(
        f"{'STT (Whisper)':<22}${cost_model['stt_per_session']:<13.4f}"
        f"${cost_model['stt_per_session']:<9.4f}${cost_model['stt_per_month']:.2f}"
    )
    lines.append(
        f"{'LLM (GPT-4o)':<22}${cost_model['llm_per_session']:<13.4f}"
        f"${cost_model['llm_per_session']:<9.4f}${cost_model['llm_per_month']:.2f}"
    )
    lines.append(
        f"{'TTS (ElevenLabs)':<22}${cost_model['tts_per_session']:<13.4f}"
        f"${cost_model['tts_per_session']:<9.4f}${cost_model['tts_per_month']:.2f}"
    )
    lines.append("─" * 21)
    lines.append(f"{'Total per user':<40}${cost_model['total_per_month']:.2f}/mo")
    lines.append(f"{'100 users':<40}${cost_model['total_per_month'] * 100:,.2f}/mo")
    lines.append(f"{'1,000 users':<40}${cost_model['total_per_month'] * 1000:,.2f}/mo")
    lines.append("─" * 55)
    return "\n".join(lines)


def render_quality_summary(quality):
    summary = quality["summary"]
    total = summary["total_scenarios"]
    pct = round(100 * summary["pass_count"] / total) if total else 0
    lines = []
    lines.append(f"Overall pass rate      : {summary['pass_count']}/{total} ({pct}%)")
    lines.append(
        f"Most-failed dimension  : {quality['worst_dimension']} "
        f"({quality['worst_dimension_fail_count']}/{total} scenarios failed it)"
    )
    lang, persona = quality["worst_group"]
    lines.append(
        f"Lowest-scoring segment : {lang}/{persona} ({quality['worst_group_avg']:.0%} avg score)"
    )
    if summary["risk_gate_pass"]:
        lines.append(f"Risk gate              : PASS ✅ ({summary['risk_correct']}/{summary['risk_total']})")
    else:
        lines.append(
            f"Risk gate              : FAIL ❌ ({summary['risk_correct']}/{summary['risk_total']}) "
            "— highlighted, do not ship"
        )
    return "\n".join(lines)


def render_markdown(cost_model, quality, observations, source_file):
    md = []
    md.append("# Voice Agent Prototype — Cost & Quality Report\n")
    md.append(f"_Source: `{os.path.relpath(source_file, PROJECT_ROOT)}`_\n")

    md.append("## 1. Cost model — 10 min/day per user\n")
    md.append(
        f"Assumptions: {SESSION_MINUTES:.0f} minutes of user speech per day, "
        f"~{TURN_SECONDS:.0f}s per conversational turn (~{TURNS_PER_SESSION:.0f} turns/session). "
        "LLM and TTS per-turn costs are averaged from the eval run.\n"
    )
    md.append("| | Per session | Per day | Per month |")
    md.append("|---|---|---|---|")
    md.append(
        f"| STT (Whisper) | ${cost_model['stt_per_session']:.4f} | "
        f"${cost_model['stt_per_session']:.4f} | ${cost_model['stt_per_month']:.2f} |"
    )
    md.append(
        f"| LLM (GPT-4o) | ${cost_model['llm_per_session']:.4f} | "
        f"${cost_model['llm_per_session']:.4f} | ${cost_model['llm_per_month']:.2f} |"
    )
    md.append(
        f"| TTS (ElevenLabs) | ${cost_model['tts_per_session']:.4f} | "
        f"${cost_model['tts_per_session']:.4f} | ${cost_model['tts_per_month']:.2f} |"
    )
    md.append(f"| **Total per user / month** | | | **${cost_model['total_per_month']:.2f}** |")
    md.append(f"| **100 users / month** | | | **${cost_model['total_per_month'] * 100:,.2f}** |")
    md.append(f"| **1,000 users / month** | | | **${cost_model['total_per_month'] * 1000:,.2f}** |\n")

    md.append("## 2. Quality summary\n")
    summary = quality["summary"]
    total = summary["total_scenarios"]
    pct = round(100 * summary["pass_count"] / total) if total else 0
    md.append(f"- Overall pass rate: **{summary['pass_count']}/{total} ({pct}%)**")
    md.append(
        f"- Most-failed dimension: **{quality['worst_dimension']}** "
        f"({quality['worst_dimension_fail_count']}/{total} scenarios failed it)"
    )
    lang, persona = quality["worst_group"]
    md.append(f"- Lowest-scoring segment: **{lang}/{persona}** ({quality['worst_group_avg']:.0%} avg score)")
    if summary["risk_gate_pass"]:
        md.append(f"- Risk gate: **PASS** ✅ ({summary['risk_correct']}/{summary['risk_total']})")
    else:
        md.append(
            f"- **Risk gate: FAIL ❌ ({summary['risk_correct']}/{summary['risk_total']})** "
            "— do not use this prompt in production\n"
        )
    md.append("")

    md.append("## 3. Observations and flags\n")
    for obs in observations:
        md.append(f"- {obs}")
    md.append("")

    return "\n".join(md)


def main():
    latest_path = find_latest_results()
    payload = load_results(latest_path)
    results = payload["results"]

    cost_model = compute_cost_model(results)
    quality = compute_quality_summary(payload, results)
    observations = generate_observations(cost_model, quality, results)

    print(render_cost_table(cost_model))
    print()
    print("─" * 55)
    print("QUALITY SUMMARY")
    print("─" * 55)
    print(render_quality_summary(quality))
    print("─" * 55)
    print()
    print("OBSERVATIONS")
    for obs in observations:
        print(f"- {obs}")

    markdown = render_markdown(cost_model, quality, observations, latest_path)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"\nSaved report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
