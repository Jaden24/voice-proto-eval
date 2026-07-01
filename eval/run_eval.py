"""Runs all scenarios in inputs/scenarios.json through the LLM (text only, no STT/TTS)
and scores each response with the LLM-as-judge. Prints a results table and summary,
saves full results to eval/results/YYYY-MM-DD_HH-MM.json.

Usage:
    python eval/run_eval.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table

from core import llm
from judge import score_response

SCENARIOS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inputs", "scenarios.json")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

DIMENSION_KEYS = ["empathy_first", "action_included", "no_guilt", "language_match"]
DIMENSION_LABELS = {
    "empathy_first": "Empathy first",
    "action_included": "Action included",
    "no_guilt": "No guilt language",
    "language_match": "Language match",
}


def load_scenarios():
    with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_all(scenarios):
    results = []
    for scenario in scenarios:
        llm_result = llm.get_response(scenario["input_text"], language_hint=scenario["language"])
        judge_result = score_response(scenario, llm_result)
        results.append({"scenario": scenario, "llm_result": llm_result, "judge_result": judge_result})
    return results


def render_table(console, results):
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID")
    table.add_column("Lang")
    table.add_column("Persona")
    table.add_column("Emp", justify="center")
    table.add_column("Act", justify="center")
    table.add_column("NoGuilt", justify="center")
    table.add_column("Lang", justify="center")
    table.add_column("Risk", justify="center")
    table.add_column("Score", justify="center")

    for r in results:
        s = r["scenario"]
        j = r["judge_result"]
        scores = j["scores"]
        risk_cell = "-" if scores["risk_handled"] == "N/A" else str(scores["risk_handled"])
        table.add_row(
            s["id"],
            s["language"],
            s["persona"],
            str(scores["empathy_first"]),
            str(scores["action_included"]),
            str(scores["no_guilt"]),
            str(scores["language_match"]),
            risk_cell,
            f"{j['total']}/{j['max']}",
        )
    console.print(table)


def compute_summary(results):
    total_scenarios = len(results)
    pass_count = sum(1 for r in results if r["judge_result"]["pass"])

    by_dimension = {k: 0 for k in DIMENSION_KEYS}
    for r in results:
        scores = r["judge_result"]["scores"]
        for k in DIMENSION_KEYS:
            by_dimension[k] += scores[k]

    risk_scenarios = [r for r in results if r["scenario"]["expected"]["risk_flag"]]
    risk_correct = sum(1 for r in risk_scenarios if r["llm_result"]["risk_flag"] is True)
    risk_total = len(risk_scenarios)
    risk_gate_pass = risk_correct == risk_total

    total_cost = sum(r["llm_result"]["cost_usd"] + r["judge_result"]["cost_usd"] for r in results)

    return {
        "total_scenarios": total_scenarios,
        "pass_count": pass_count,
        "by_dimension": by_dimension,
        "risk_correct": risk_correct,
        "risk_total": risk_total,
        "risk_gate_pass": risk_gate_pass,
        "total_cost": total_cost,
    }


def print_summary(summary):
    bar = "─" * 43
    total = summary["total_scenarios"]
    print()
    print(bar)
    print("EVAL SUMMARY")
    print(bar)
    pct = round(100 * summary["pass_count"] / total) if total else 0
    print(f"Total scenarios     : {total}")
    print(f"Overall pass rate   : {summary['pass_count']}/{total} ({pct}%)")
    print()
    print("By dimension:")
    for k in DIMENSION_KEYS:
        print(f"  {DIMENSION_LABELS[k]:<18}: {summary['by_dimension'][k]}/{total}")
    print()
    print(f"RISK GATE: {summary['risk_correct']}/{summary['risk_total']} crisis cases handled correctly")
    if summary["risk_gate_pass"]:
        print("→ PASS ✅")
    else:
        print("→ FAIL ❌ (do not use this prompt in production)")
    print()
    print(f"Total eval cost     : ${summary['total_cost']:.4f}  (LLM calls + judge calls)")
    print(bar)


def save_results(results, summary, timestamp):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{timestamp}.json")
    payload = {
        "timestamp": timestamp,
        "summary": summary,
        "results": [
            {
                "id": r["scenario"]["id"],
                "language": r["scenario"]["language"],
                "persona": r["scenario"]["persona"],
                "input_text": r["scenario"]["input_text"],
                "expected": r["scenario"]["expected"],
                "llm_result": r["llm_result"],
                "judge_result": r["judge_result"],
            }
            for r in results
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def main():
    console = Console()
    scenarios = load_scenarios()
    console.print(f"Running {len(scenarios)} scenarios through the LLM...\n")
    results = run_all(scenarios)

    render_table(console, results)
    summary = compute_summary(results)
    print_summary(summary)

    timestamp = time.strftime("%Y-%m-%d_%H-%M")
    out_path = save_results(results, summary, timestamp)
    console.print(f"\nSaved full results to {out_path}")

    if not summary["risk_gate_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
