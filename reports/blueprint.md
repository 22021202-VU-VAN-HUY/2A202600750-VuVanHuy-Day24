# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Vũ Văn Huy  
**Mã học viên:** 2A202600750  
**Ngay:** 2026-06-30

## Guard Stack Architecture

```text
User Input
  -> Presidio/Regex PII Scan (~9.81ms P95)
  -> NeMo Input Rail or fallback heuristic (~0.63ms P95)
  -> Day 18 RAG Pipeline
  -> Output Rail / sensitive-output check
  -> User Response
```

## Latency Budget

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---:|---:|---:|---:|
| Presidio PII | 7.66 | 9.81 | 10.01 | <10ms |
| NeMo Input Rail / fallback | 0.46 | 0.63 | 0.85 | <300ms |
| RAG Pipeline | not benchmarked here | not benchmarked here | not benchmarked here | <2000ms |
| NeMo Output Rail / fallback | not benchmarked here | not benchmarked here | not benchmarked here | <300ms |
| **Total Guard** | **8.16** | **10.25** | **10.45** | **<500ms** |

**Budget OK?** Yes  
**Comment:** Local regex/fallback rails are comfortably below budget. If real NeMo LLM rails are enabled, re-run latency because network/model latency will dominate.

## CI/CD Gates

```yaml
name: RAG Eval Gate
on:
  pull_request:
    branches: [main]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: python setup_answers.py
        env:
          MIMO_API_KEY: ${{ secrets.MIMO_API_KEY }}
          MIMO_BASE_URL: https://api.xiaomimimo.com/v1
          MIMO_MODEL: mimo-v2.5-pro
      - run: python src/phase_a_ragas.py
      - run: pytest tests/ -q
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: lab24-reports
          path: reports/
```

Required gates before merge:

- RAGAS faithfulness >= 0.75 on the 50-question set, or explicit waiver with failure analysis.
- Adversarial suite pass rate >= 90% for production target; current lab result is 20/20.
- Guard P95 latency < 500ms; current measured value is 10.25ms.
- No `# TODO` markers in `src/phase_*.py`.

## Monitoring Dashboard

| Metric | Current Lab Value | Alert Threshold | Action |
|---|---:|---:|---|
| RAGAS avg_score | 0.419 | <0.65 | Review retrieval, reranking, and prompt grounding |
| Worst RAGAS metric | context_precision | <0.60 | Tune reranker and metadata/version filters |
| Dominant failure distribution | factual | spike vs baseline | Inspect direct-lookup retrieval noise |
| Adversarial pass rate | 20/20 | <18/20 | Add new attack patterns to rails |
| Guard P95 latency | 10.25ms | >500ms | Check NeMo/API latency and fall back if needed |

## Actual Lab Results

| Item | Result |
|---|---:|
| RAGAS avg_score (50q) | 0.419 |
| Worst metric | context_precision |
| Dominant failure distribution | factual |
| Cohen's kappa | 1.000 |
| Adversarial pass rate | 20 / 20 |
| Guard P95 latency | 10.25 ms |

## Improvement Plan

The main RAG weakness is retrieval precision. The next production iteration should add stronger reranking, policy-version metadata filters, and query expansion for Vietnamese HR terms. Guardrail performance is strong in the lab suite, but the Windows setup uses fallback rails instead of full NeMo because the latest NeMo dependency chain requires native C++ build tooling. Before production, enable real NeMo rails in Linux CI and re-run adversarial plus latency benchmarks.
