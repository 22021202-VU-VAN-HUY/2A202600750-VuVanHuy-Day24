from __future__ import annotations

"""Phase C: production guardrails with PII scan, input/output rails, and latency."""

import asyncio
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE

PII_PATTERNS = {
    "VN_CCCD": re.compile(r"\b(?:\d{12}|\d{9})\b"),
    "VN_PHONE": re.compile(r"\b0[3-9]\d{8}\b"),
    "EMAIL_ADDRESS": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
}

BLOCK_PATTERNS = [
    r"ignore (your|all|previous)|b[oỏ] qua",
    r"system override|admin command|priority 1|dump all|training data",
    r"\bdan\b|do anything now|unrestricted",
    r"confidential|employee records|all employee|b[aả]ng l[uư][oơ]ng|l[uư][oơ]ng.*nh[aâ]n vi[eê]n",
    r"cccd|cmnd|s[oố] ?[dđ]i[eệ]n tho[aạ]i|email.*nh[aâ]n vi[eê]n",
    r"prompt|system instructions|context",
    r"t[aấ]n c[oô]ng|attack|harmful|illegal",
]

HR_TERMS = [
    "nghi", "phep", "luong", "thuong", "bao hiem", "phu cap", "tam ung",
    "cong tac", "vpn", "mat khau", "dao tao", "mentor", "thu viec", "hr",
]

OFF_TOPIC_TERMS = [
    "bai tho", "mua thu", "nau", "pho", "bitcoin", "ethereum", "phim",
    "marvel", "phuong trinh", "vi phan",
]


def _normalize(text: str) -> str:
    text = (text or "").lower()
    replacements = {
        "â": "a", "ă": "a", "á": "a", "à": "a", "ả": "a", "ã": "a", "ạ": "a",
        "ấ": "a", "ầ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a",
        "ắ": "a", "ằ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a",
        "ê": "e", "é": "e", "è": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
        "ế": "e", "ề": "e", "ể": "e", "ễ": "e", "ệ": "e",
        "ô": "o", "ơ": "o", "ó": "o", "ò": "o", "ỏ": "o", "õ": "o", "ọ": "o",
        "ố": "o", "ồ": "o", "ổ": "o", "ỗ": "o", "ộ": "o",
        "ớ": "o", "ờ": "o", "ở": "o", "ỡ": "o", "ợ": "o",
        "ư": "u", "ú": "u", "ù": "u", "ủ": "u", "ũ": "u", "ụ": "u",
        "ứ": "u", "ừ": "u", "ử": "u", "ữ": "u", "ự": "u",
        "í": "i", "ì": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
        "ý": "y", "ỳ": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
        "đ": "d",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def setup_presidio():
    """Create a Presidio analyzer/anonymizer with Vietnamese regex recognizers."""
    try:
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
        from presidio_anonymizer import AnonymizerEngine

        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()
        registry.add_recognizer(
            PatternRecognizer(
                supported_entity="VN_CCCD",
                patterns=[
                    Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
                    Pattern("CMND 9 digits", r"\b\d{9}\b", 0.7),
                ],
            )
        )
        registry.add_recognizer(
            PatternRecognizer(
                supported_entity="VN_PHONE",
                patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
            )
        )
        return AnalyzerEngine(registry=registry), AnonymizerEngine()
    except Exception:
        return None, None


def _regex_pii_scan(text: str) -> dict:
    entities = []
    anonymized = text
    for entity_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append(
                {
                    "type": entity_type,
                    "text": match.group(0),
                    "score": 0.9,
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    for entity in sorted(entities, key=lambda item: item["start"], reverse=True):
        anonymized = anonymized[: entity["start"]] + f"<{entity['type']}>" + anonymized[entity["end"] :]
    return {"has_pii": bool(entities), "entities": entities, "anonymized": anonymized}


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Detect and anonymize CCCD/CMND, Vietnamese phone numbers, and email."""
    if analyzer is None or anonymizer is None:
        return _regex_pii_scan(text)

    try:
        results = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)
        regex_result = _regex_pii_scan(text)
        seen = {(r.entity_type, r.start, r.end) for r in results}
        for entity in regex_result["entities"]:
            key = (entity["type"], entity["start"], entity["end"])
            if key not in seen:
                # Keep custom regex hits even if the local spaCy model is absent or weak.
                pass
        if not results:
            return regex_result
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results).text
        entities = [
            {
                "type": r.entity_type,
                "text": text[r.start : r.end],
                "score": round(float(r.score), 3),
                "start": r.start,
                "end": r.end,
            }
            for r in results
        ]
        merged = {(_e["type"], _e["start"], _e["end"]): _e for _e in entities + regex_result["entities"]}
        if regex_result["has_pii"]:
            anonymized = regex_result["anonymized"]
        return {"has_pii": bool(merged), "entities": list(merged.values()), "anonymized": anonymized}
    except Exception:
        return _regex_pii_scan(text)


def setup_nemo_rails():
    """Load NeMo Guardrails when installed; otherwise return None for fallback rails."""
    try:
        from nemoguardrails import LLMRails, RailsConfig

        config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
        return LLMRails(config)
    except Exception:
        return None


def _heuristic_input_block(text: str) -> tuple[bool, str | None]:
    normalized = _normalize(text)
    if any(re.search(pattern, normalized) for pattern in BLOCK_PATTERNS):
        return True, "policy_or_injection"
    if any(term in normalized for term in OFF_TOPIC_TERMS):
        return True, "off_topic"
    if not any(term in normalized for term in HR_TERMS):
        return True, "off_topic"
    return False, None


async def check_input_rail(text: str, rails=None) -> dict:
    """Check topic, jailbreak, prompt-injection, and PII-request input rails."""
    if rails is None:
        rails = setup_nemo_rails()
    if rails is not None:
        try:
            response = await rails.generate_async(messages=[{"role": "user", "content": text}])
            response_text = response if isinstance(response, str) else str(response)
            blocked = any(keyword in response_text.lower() for keyword in ["xin loi", "khong the", "cannot", "sorry"])
            return {
                "allowed": not blocked,
                "blocked_reason": "nemo_input_rail" if blocked else None,
                "response": response_text,
            }
        except Exception:
            pass

    blocked, reason = _heuristic_input_block(text)
    return {
        "allowed": not blocked,
        "blocked_reason": reason if blocked else None,
        "response": "blocked by fallback rail" if blocked else "allowed by fallback rail",
    }


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Check the assistant output before returning it to the user."""
    pii_result = pii_scan(answer)
    normalized = _normalize(answer)
    sensitive = pii_result["has_pii"] or any(
        term in normalized for term in ["confidential", "system prompt", "employee records", "mat khau admin"]
    )
    if sensitive:
        return {
            "safe": False,
            "flagged_reason": "sensitive_output",
            "final_answer": "Xin loi, cau tra loi co thong tin nhay cam nen da bi chan.",
        }
    return {"safe": True, "flagged_reason": None, "final_answer": answer}


def run_adversarial_suite(adversarial_set: list[dict], rails=None, analyzer=None, anonymizer=None) -> list[dict]:
    """Run all adversarial inputs through PII and input rails."""

    async def _run_all() -> list[dict]:
        results = []
        for item in adversarial_set:
            blocked_by = None
            pii_result = pii_scan(item["input"], analyzer, anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"

            if blocked_by is None:
                rail_result = await check_input_rail(item["input"], rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            results.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "input": item["input"][:120],
                    "expected": item["expected"],
                    "actual": actual,
                    "blocked_by": blocked_by,
                    "passed": actual == item["expected"],
                }
            )
        return results

    results = asyncio.run(_run_all())
    passed = sum(1 for result in results if result["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    values = sorted(values)
    n = len(values)

    def pick(pct: float) -> float:
        index = min(n - 1, max(0, int(round((n - 1) * pct))))
        return round(values[index], 2)

    return {"p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99)}


def measure_p95_latency(test_inputs: list[str], n_runs: int = 20, rails=None, analyzer=None, anonymizer=None) -> dict:
    """Measure P50/P95/P99 latency for PII and input-rail layers."""
    presidio_times: list[float] = []
    nemo_times: list[float] = []
    total_times: list[float] = []
    inputs = (test_inputs or ["test"])[:n_runs]

    async def _measure() -> None:
        for text in inputs:
            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - t1) * 1000

            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    asyncio.run(_measure())
    total_p = _percentiles(total_times)
    return {
        "presidio_ms": _percentiles(presidio_times),
        "nemo_ms": _percentiles(nemo_times),
        "total_ms": total_p,
        "latency_budget_ok": total_p["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


def save_guard_report(results: list[dict], latency: dict, path: str = "reports/guard_results.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    passed = sum(1 for result in results if result["passed"])
    payload = {
        "total": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 3) if results else 0.0,
        "results": results,
        "latency": latency,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Phase C report saved -> {path}")


if __name__ == "__main__":
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    analyzer, anonymizer = setup_presidio()
    rails = setup_nemo_rails()
    results = run_adversarial_suite(adversarial_set, rails=rails, analyzer=analyzer, anonymizer=anonymizer)
    latency = measure_p95_latency([item["input"] for item in adversarial_set], n_runs=20, rails=rails, analyzer=analyzer, anonymizer=anonymizer)
    save_guard_report(results, latency)
