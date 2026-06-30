# LLM Judge Bias Report - Phase B

**Sinh viên:** Vũ Văn Huy  
**Mã học viên:** 2A202600750  
**Ngay:** 2026-06-30  
**Judge model:** mimo-v2.5-pro, deterministic fallback by default

## 1. Pairwise Judge Results

| # | Question summary | Winner | Reasoning summary |
|---:|---|---|---|
| 1 | Annual leave entitlement | A | A is more specific and references the current 2024 policy |

## 2. Swap-and-Average Results

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---:|---|---|---|---|
| 1 | A | A | A | True |

**Position bias rate:** 0.0% (= 0 inconsistent / 1 total)

## 3. Cohen's Kappa Analysis

The generated report uses the provided 10 human labels as a calibration smoke test and maps the deterministic judge labels to the same binary labels.

| Metric | Value |
|---|---:|
| Human labels | 10 |
| Judge labels | 10 |
| Cohen's kappa | 1.000 |
| Interpretation | almost perfect |

In a production run, this should be recomputed on real judge outputs for the same 10 labeled examples. The current value mainly verifies the implementation path and report structure.

## 4. Verbosity Bias

In decisive cases:

| Measure | Value |
|---|---:|
| A wins + A longer than B | 1 |
| B wins + B longer than A | 0 |
| Total decisive | 1 |
| Verbosity bias rate | 100.0% |

Because the sample has only one decisive pair, the verbosity-bias number is not statistically meaningful. It is still useful as a warning that longer answers can win when they also include newer policy/version evidence.

## 5. General Notes

Swap-and-average is enabled and produced consistent winners on the demo pair. Position bias is low in the current sample, but the sample is intentionally small to keep the lab run cheap and deterministic. For production, use `LAB24_USE_LLM_JUDGE=1`, run at least 30-50 answer pairs, and keep the deterministic fallback for CI smoke tests.
