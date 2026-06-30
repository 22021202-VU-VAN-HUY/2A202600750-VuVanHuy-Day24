from __future__ import annotations

"""Phase A: RAGAS production evaluation for the 50-question lab set."""

import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ANSWERS_PATH, TEST_SET_PATH

Distribution = str

DIAGNOSTIC_TREE = {
    "faithfulness": ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy": ("Answer does not match question", "Improve prompt template"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (
            self.faithfulness
            + self.answer_relevancy
            + self.context_precision
            + self.context_recall
        ) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }
        return min(scores, key=scores.get)


def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    """Load the 50-question test set."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    """Load generated answers from setup_answers.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"answers_50q.json not found at {path}\n"
            "Run first: python setup_answers.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    """Group questions into factual, multi_hop, and adversarial buckets."""
    groups = {"factual": [], "multi_hop": [], "adversarial": []}
    for item in test_set:
        dist = item.get("distribution")
        if dist not in groups:
            raise ValueError(f"Unknown distribution: {dist!r}")
        groups[dist].append(item)
    return groups


def run_ragas_50q(answers: list[dict]) -> list[RagasResult]:
    """Run the Day 18 RAGAS-compatible evaluator over all answers."""
    from src.m4_eval import evaluate_ragas

    questions = [a.get("question", "") for a in answers]
    ans_texts = [a.get("answer", "") for a in answers]
    contexts = [a.get("contexts", []) for a in answers]
    ground_truths = [a.get("ground_truth", "") for a in answers]

    raw = evaluate_ragas(questions, ans_texts, contexts, ground_truths)
    per_question = raw.get("per_question", [])

    results: list[RagasResult] = []
    for a, pq in zip(answers, per_question):
        getter = pq.get if isinstance(pq, dict) else lambda key, default=0.0: getattr(pq, key, default)
        results.append(
            RagasResult(
                question_id=int(a.get("id", a.get("question_id", len(results) + 1))),
                distribution=a.get("distribution", "factual"),
                question=a.get("question", ""),
                answer=a.get("answer", ""),
                contexts=a.get("contexts", []),
                ground_truth=a.get("ground_truth", ""),
                faithfulness=float(getter("faithfulness", 0.0)),
                answer_relevancy=float(getter("answer_relevancy", 0.0)),
                context_precision=float(getter("context_precision", 0.0)),
                context_recall=float(getter("context_recall", 0.0)),
            )
        )
    return results


def bottom_10(results: list[RagasResult]) -> list[dict]:
    """Return the ten lowest average-score questions with diagnosis fields."""
    output = []
    for rank, result in enumerate(sorted(results, key=lambda r: r.avg_score)[:10], start=1):
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[result.worst_metric]
        output.append(
            {
                "rank": rank,
                "question_id": result.question_id,
                "distribution": result.distribution,
                "question": result.question,
                "avg_score": round(result.avg_score, 4),
                "worst_metric": result.worst_metric,
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        )
    return output


def cluster_analysis(results: list[RagasResult]) -> dict:
    """Build a worst_metric by distribution matrix and summarize it."""
    matrix = {
        metric: {"factual": 0, "multi_hop": 0, "adversarial": 0}
        for metric in DIAGNOSTIC_TREE
    }
    for result in results:
        if result.distribution in matrix[result.worst_metric]:
            matrix[result.worst_metric][result.distribution] += 1

    distributions = ["factual", "multi_hop", "adversarial"]
    dominant_dist = max(distributions, key=lambda d: sum(row[d] for row in matrix.values()))
    dominant_metric = max(matrix, key=lambda m: sum(matrix[m].values()))
    _, fix = DIAGNOSTIC_TREE[dominant_metric]
    insight = (
        f"Distribution '{dominant_dist}' has the most low-score cases, while "
        f"'{dominant_metric}' is the most frequent weakest metric. Suggested next step: {fix}."
    )

    return {
        "matrix": matrix,
        "dominant_failure_distribution": dominant_dist,
        "dominant_failure_metric": dominant_metric,
        "insight": insight,
    }


def save_phase_a_report(
    results: list[RagasResult],
    clusters: dict,
    path: str = "reports/ragas_50q.json",
) -> None:
    """Save the Phase A JSON report."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    per_dist: dict[str, dict] = {}
    for dist in ["factual", "multi_hop", "adversarial"]:
        subset = [r for r in results if r.distribution == dist]
        if subset:
            per_dist[dist] = {
                "count": len(subset),
                "faithfulness": sum(r.faithfulness for r in subset) / len(subset),
                "answer_relevancy": sum(r.answer_relevancy for r in subset) / len(subset),
                "context_precision": sum(r.context_precision for r in subset) / len(subset),
                "context_recall": sum(r.context_recall for r in subset) / len(subset),
                "avg_score": sum(r.avg_score for r in subset) / len(subset),
            }

    report = {
        "total_questions": len(results),
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        "bottom_10": bottom_10(results),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase A report saved -> {path}")


if __name__ == "__main__":
    test_set = load_test_set_50q()
    print(f"Loaded {len(test_set)} questions")

    groups = group_by_distribution(test_set)
    for dist, questions in groups.items():
        print(f"  {dist}: {len(questions)} questions")

    answers = load_answers()
    results = run_ragas_50q(answers)

    if results:
        clusters = cluster_analysis(results)
        save_phase_a_report(results, clusters)
        print("\nBottom 10 worst questions:")
        for item in bottom_10(results):
            print(
                f"  #{item['rank']} [{item['distribution']}] "
                f"{item['question'][:50]}... avg={item['avg_score']:.3f} "
                f"worst={item['worst_metric']}"
            )
        print(
            f"\nDominant failure: {clusters.get('dominant_failure_distribution')} / "
            f"{clusters.get('dominant_failure_metric')}"
        )
    else:
        print("No results.")
