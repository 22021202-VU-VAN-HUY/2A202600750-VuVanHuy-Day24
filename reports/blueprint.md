# Blueprint CI/CD: RAG Eval + Guardrail Stack

**Sinh viên:** Vũ Văn Huy  
**Mã học viên:** 2A202600750  
**Ngày:** 2026-06-30

## Kiến Trúc Guard Stack

```text
User Input
  -> Presidio/Regex PII Scan (~9.81ms P95)
  -> NeMo Input Rail hoặc fallback heuristic (~0.63ms P95)
  -> Day 18 RAG Pipeline
  -> Output Rail / kiểm tra output nhạy cảm
  -> User Response
```

Luồng bảo vệ gồm nhiều lớp. Lớp đầu phát hiện PII như CCCD, CMND, số điện thoại Việt Nam và email. Lớp tiếp theo chặn jailbreak, prompt injection, yêu cầu dữ liệu nhân viên và câu hỏi ngoài phạm vi HR. Sau khi RAG sinh câu trả lời, output rail kiểm tra lại để tránh trả về thông tin nhạy cảm.

## Ngân Sách Độ Trễ

| Lớp | P50 (ms) | P95 (ms) | P99 (ms) | Ngân sách |
|---|---:|---:|---:|---:|
| Presidio PII | 7.66 | 9.81 | 10.01 | <10ms |
| NeMo Input Rail / fallback | 0.46 | 0.63 | 0.85 | <300ms |
| RAG Pipeline | chưa benchmark trong báo cáo này | chưa benchmark trong báo cáo này | chưa benchmark trong báo cáo này | <2000ms |
| NeMo Output Rail / fallback | chưa benchmark trong báo cáo này | chưa benchmark trong báo cáo này | chưa benchmark trong báo cáo này | <300ms |
| **Tổng guard** | **8.16** | **10.25** | **10.45** | **<500ms** |

**Đạt ngân sách?** Có  
**Nhận xét:** Regex/fallback rail chạy local nên nhanh hơn nhiều so với ngân sách. Nếu bật NeMo rail thật qua LLM/API, cần đo lại latency vì độ trễ mạng và model sẽ chiếm phần lớn thời gian.

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

Các điều kiện nên chặn merge:

- RAGAS faithfulness >= 0.75 trên bộ 50 câu hỏi, hoặc phải có ghi chú miễn trừ kèm phân tích lỗi.
- Adversarial suite pass rate >= 90% cho mục tiêu production; kết quả lab hiện tại là 20/20.
- Guard P95 latency < 500ms; kết quả đo hiện tại là 10.25ms.
- Không còn `# TODO` trong `src/phase_*.py`.

## Monitoring Dashboard

| Chỉ số | Giá trị lab hiện tại | Ngưỡng cảnh báo | Hành động |
|---|---:|---:|---|
| RAGAS avg_score | 0.419 | <0.65 | Kiểm tra retrieval, reranking và prompt grounding |
| Metric RAGAS yếu nhất | context_precision | <0.60 | Tinh chỉnh reranker và bộ lọc metadata/version |
| Nhóm lỗi chủ đạo | factual | tăng bất thường so với baseline | Kiểm tra nhiễu retrieval ở câu hỏi tra cứu trực tiếp |
| Adversarial pass rate | 20/20 | <18/20 | Bổ sung attack patterns mới vào rails |
| Guard P95 latency | 10.25ms | >500ms | Kiểm tra NeMo/API latency và bật fallback nếu cần |

## Kết Quả Thực Tế Từ Lab

| Hạng mục | Kết quả |
|---|---:|
| RAGAS avg_score (50q) | 0.419 |
| Metric yếu nhất | context_precision |
| Nhóm lỗi chủ đạo | factual |
| Cohen's kappa | 1.000 |
| Adversarial pass rate | 20 / 20 |
| Guard P95 latency | 10.25 ms |

## Kế Hoạch Cải Thiện

Điểm yếu chính của RAG hiện tại là độ chính xác của retrieval. Vòng cải thiện tiếp theo nên tập trung vào reranking mạnh hơn, bộ lọc metadata theo phiên bản chính sách, và query expansion cho thuật ngữ HR tiếng Việt. Guardrail đạt kết quả tốt trên bộ adversarial lab, nhưng môi trường Windows đang dùng fallback rail thay vì NeMo đầy đủ vì chuỗi dependency mới nhất của NeMo cần native C++ build tooling. Trước khi đưa vào production, nên bật NeMo rail thật trong Linux CI rồi chạy lại benchmark adversarial và latency.
