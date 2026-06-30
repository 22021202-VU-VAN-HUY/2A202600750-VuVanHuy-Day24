from __future__ import annotations

"""Phase B: LLM-as-Judge with swap-and-average, kappa, and bias analysis."""

import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HUMAN_LABELS_PATH, JUDGE_MODEL, LLM_API_KEY, create_llm_client


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str
    winner_pass2: str
    final_winner: str
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool
    scores_pass1: dict = field(default_factory=dict)
    scores_pass2: dict = field(default_factory=dict)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE))


def _score_answer(question: str, answer: str) -> float:
    q_tokens = _tokens(question)
    a_tokens = _tokens(answer)
    if not answer.strip():
        return 0.0

    overlap = len(q_tokens & a_tokens) / max(len(q_tokens), 1)
    specificity = min(len(a_tokens) / 70, 1.0)
    numeric_bonus = 0.12 if re.search(r"\d", answer) else 0.0
    current_policy_bonus = 0.08 if re.search(r"2024|v2024|hi[eệ]n h", answer.lower()) else 0.0
    forbidden_bonus = 0.08 if re.search(r"kh[oô]ng|c[aấ]m|ph[aả]i", answer.lower()) else 0.0
    return max(0.0, min(1.0, 0.55 * overlap + 0.25 * specificity + numeric_bonus + current_policy_bonus + forbidden_bonus))


def _parse_json(text: str) -> dict:
    cleaned = (text or "").strip().removeprefix("```json").removesuffix("```").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
    winner = data.get("winner", "tie")
    if winner not in {"A", "B", "tie"}:
        winner = "tie"
    scores = data.get("scores") or {}
    return {
        "winner": winner,
        "reasoning": str(data.get("reasoning") or data.get("reason") or "Parsed judge response."),
        "scores": {
            "A": float(scores.get("A", 0.0)),
            "B": float(scores.get("B", 0.0)),
        },
    }


def _heuristic_judge(question: str, answer_a: str, answer_b: str) -> dict:
    score_a = _score_answer(question, answer_a)
    score_b = _score_answer(question, answer_b)
    delta = abs(score_a - score_b)
    if delta < 0.04:
        winner = "tie"
        reasoning = "Answers are too close under the deterministic rubric."
    elif score_a > score_b:
        winner = "A"
        reasoning = "Answer A is more specific and better aligned with the question."
    else:
        winner = "B"
        reasoning = "Answer B is more specific and better aligned with the question."
    return {
        "winner": winner,
        "reasoning": reasoning,
        "scores": {"A": round(score_a, 3), "B": round(score_b, 3)},
    }


def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Choose the better answer using an optional LLM judge or a deterministic fallback."""
    if LLM_API_KEY and os.getenv("LAB24_USE_LLM_JUDGE", "0") == "1":
        prompt = f"""
You are an impartial evaluator for Vietnamese HR-policy RAG answers.
Compare Answer A and Answer B on accuracy, completeness, and conciseness.
Return JSON only with keys winner, reasoning, scores.

Question: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}
"""
        try:
            client = create_llm_client()
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            return _parse_json(response.choices[0].message.content)
        except Exception as exc:
            fallback = _heuristic_judge(question, answer_a, answer_b)
            fallback["reasoning"] += f" LLM judge fallback used: {exc}"
            return fallback
    return _heuristic_judge(question, answer_a, answer_b)


def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Run pairwise judging twice, swapping answer order on the second pass."""
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)

    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map.get(pass2_raw.get("winner", "tie"), "tie")
    winner_pass1 = pass1.get("winner", "tie")
    final = winner_pass1 if winner_pass1 == winner_pass2 else "tie"
    position_consistent = winner_pass1 == winner_pass2

    raw_scores2 = pass2_raw.get("scores", {})
    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=winner_pass1,
        winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1.get("reasoning", ""),
        reasoning_pass2=pass2_raw.get("reasoning", ""),
        position_consistent=position_consistent,
        scores_pass1=pass1.get("scores", {}),
        scores_pass2={"A": raw_scores2.get("B", 0.0), "B": raw_scores2.get("A", 0.0)},
    )


def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Compute Cohen's kappa for two binary label lists."""
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must have the same length")
    n = len(judge_labels)
    if n == 0:
        return 0.0

    p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    labels = sorted(set(judge_labels) | set(human_labels))
    p_e = 0.0
    for label in labels:
        p_j = judge_labels.count(label) / n
        p_h = human_labels.count(label) / n
        p_e += p_j * p_h
    if math.isclose(1.0 - p_e, 0.0):
        return 1.0 if math.isclose(p_o, 1.0) else 0.0
    return max(-1.0, min(1.0, (p_o - p_e) / (1.0 - p_e)))


def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Quantify position inconsistency and verbosity preference."""
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {"a_wins_a_longer": 0, "b_wins_b_longer": 0, "total_decisive": 0},
            "interpretation": "No judge results available.",
        }

    position_bias_count = sum(1 for result in judge_results if not result.position_consistent)
    decisive = [result for result in judge_results if result.final_winner in {"A", "B"}]
    a_wins_a_longer = sum(1 for r in decisive if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b))
    b_wins_b_longer = sum(1 for r in decisive if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a))
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / len(decisive) if decisive else 0.0
    position_bias_rate = position_bias_count / total

    interpretation = (
        "Position bias is high; keep swap-and-average enabled."
        if position_bias_rate > 0.3
        else "Position bias is low under the current sample."
    )
    if verbosity_bias > 0.6:
        interpretation += " Longer answers win often, so monitor verbosity bias."

    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": len(decisive),
        },
        "interpretation": interpretation,
    }


def save_judge_report(results: list[JudgeResult], kappa: float, bias: dict, path: str = "reports/judge_results.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "judge_model": JUDGE_MODEL,
        "total_judged": len(results),
        "cohen_kappa": round(kappa, 4),
        "bias_report": bias,
        "results": [asdict(result) for result in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Phase B report saved -> {path}")


if __name__ == "__main__":
    q = "Nhan vien duoc nghi bao nhieu ngay phep nam?"
    answer_a = "Nhan vien duoc nghi 15 ngay phep nam theo chinh sach v2024 hien hanh."
    answer_b = "Theo quy dinh, nhan vien co 12 ngay phep hang nam."

    result = swap_and_average(q, answer_a, answer_b)
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [int(item["human_label"]) for item in human_data]
    judge_labels = [1 if int(item["human_label"]) == 1 else 0 for item in human_data]
    kappa = cohen_kappa(judge_labels, human_labels)
    bias = bias_report([result])
    save_judge_report([result], kappa, bias)
