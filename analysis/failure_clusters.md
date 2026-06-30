# Failure Cluster Analysis - Phase A

**Sinh viên:** Vũ Văn Huy  
**Mã học viên:** 2A202600750  
**Ngay:** 2026-06-30

## 1. Aggregate RAGAS Scores Theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 0.456 | 0.437 | 0.525 |
| answer_relevancy | 0.639 | 0.496 | 0.495 |
| context_precision | 0.303 | 0.285 | 0.323 |
| context_recall | 0.412 | 0.342 | 0.292 |
| **avg_score** | **0.452** | **0.390** | **0.409** |

## 2. Bottom 10 Questions

| Rank | Distribution | Question ID | avg_score | worst_metric |
|---:|---|---:|---:|---|
| 1 | factual | 5 | 0.238 | context_precision |
| 2 | multi_hop | 34 | 0.252 | faithfulness |
| 3 | multi_hop | 22 | 0.257 | context_recall |
| 4 | multi_hop | 35 | 0.269 | faithfulness |
| 5 | factual | 18 | 0.278 | context_recall |
| 6 | adversarial | 44 | 0.301 | context_precision |
| 7 | multi_hop | 25 | 0.303 | context_recall |
| 8 | factual | 8 | 0.313 | context_recall |
| 9 | adversarial | 50 | 0.314 | context_recall |
| 10 | multi_hop | 33 | 0.330 | context_recall |

## 3. Failure Cluster Matrix

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 0 | 2 | 0 | 2 |
| answer_relevancy | 0 | 0 | 0 | 0 |
| context_precision | 15 | 12 | 3 | 30 |
| context_recall | 5 | 6 | 7 | 18 |

## 4. Dominant Failure Analysis

**Dominant distribution:** factual  
**Dominant metric:** context_precision

The most frequent weak metric is context_precision, which means retrieval often returns partially related but noisy chunks. Factual questions have the highest count of weakest-metric cases because even direct lookup questions can retrieve neighboring policy sections, versioned documents, or generic HR chunks. Multi-hop questions still have the lowest average score overall, but their failures split between context recall and faithfulness rather than one single bucket.

## 5. Suggested Fixes

| Metric yeu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | Answer may combine facts not fully supported by retrieved context | Require citations from retrieved chunks and lower generation temperature |
| context_recall | Relevant evidence missing from top contexts | Increase BM25/dense top_k before rerank and add query expansion for policy terms |
| context_precision | Too many irrelevant chunks in final context | Strengthen reranking, add metadata filters for policy/version, reduce final top_k if noisy |
| answer_relevancy | Answer drifts from exact question intent | Add direct-answer prompt format and reject unsupported extra details |

## 6. Adversarial Distribution Notes

Adversarial avg_score is 0.409, lower than factual 0.452 but higher than multi_hop 0.390. This suggests version-conflict and trap questions are difficult, but multi-step retrieval/calculation is still the weakest area. Two adversarial questions appear in the bottom 10: password rotation and personal VPN usage. Both are version/specific-policy style questions where the retriever needs the exact current policy chunk, not just a semantically nearby security document.
