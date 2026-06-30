# Phân Tích Cụm Lỗi - Phase A

**Sinh viên:** Vũ Văn Huy  
**Mã học viên:** 2A202600750  
**Ngày:** 2026-06-30

## 1. Điểm RAGAS Theo Nhóm Câu Hỏi

| Chỉ số | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 0.456 | 0.437 | 0.525 |
| answer_relevancy | 0.639 | 0.496 | 0.495 |
| context_precision | 0.303 | 0.285 | 0.323 |
| context_recall | 0.412 | 0.342 | 0.292 |
| **avg_score** | **0.452** | **0.390** | **0.409** |

## 2. Bottom 10 Câu Hỏi Có Điểm Thấp Nhất

| Hạng | Nhóm | ID câu hỏi | avg_score | Chỉ số yếu nhất |
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

## 3. Ma Trận Cụm Lỗi

Mỗi ô thể hiện số câu hỏi có chỉ số yếu nhất tương ứng với từng nhóm.

| Chỉ số yếu nhất | factual | multi_hop | adversarial | Tổng |
|---|---:|---:|---:|---:|
| faithfulness | 0 | 2 | 0 | 2 |
| answer_relevancy | 0 | 0 | 0 | 0 |
| context_precision | 15 | 12 | 3 | 30 |
| context_recall | 5 | 6 | 7 | 18 |

## 4. Phân Tích Lỗi Chủ Đạo

**Nhóm lỗi chủ đạo:** factual  
**Chỉ số yếu nhất chủ đạo:** context_precision

Chỉ số yếu xuất hiện nhiều nhất là `context_precision`, nghĩa là bước retrieval thường lấy về các đoạn tài liệu có liên quan một phần nhưng còn nhiễu. Nhóm factual có số lượng lỗi cao nhất vì ngay cả câu hỏi tra cứu trực tiếp cũng có thể kéo theo các đoạn chính sách lân cận, tài liệu nhiều phiên bản, hoặc các đoạn HR quá chung chung. Nhóm multi_hop có điểm trung bình thấp nhất, nhưng lỗi của nhóm này phân tán giữa `context_recall` và `faithfulness`, thay vì tập trung vào một chỉ số duy nhất.

## 5. Hướng Cải Thiện

| Chỉ số yếu | Nguyên nhân gốc | Cách cải thiện |
|---|---|---|
| faithfulness | Câu trả lời có thể kết hợp thông tin chưa được context hỗ trợ đầy đủ | Bắt buộc trích dẫn từ context, giảm temperature, yêu cầu chỉ trả lời dựa trên tài liệu |
| context_recall | Context top-k thiếu đoạn bằng chứng quan trọng | Tăng BM25/dense top_k trước khi rerank, thêm query expansion cho thuật ngữ HR tiếng Việt |
| context_precision | Context cuối chứa quá nhiều đoạn không cần thiết | Tăng chất lượng reranking, thêm bộ lọc metadata theo loại chính sách và phiên bản, giảm final top_k nếu nhiễu |
| answer_relevancy | Câu trả lời lệch khỏi đúng ý hỏi | Chuẩn hóa prompt trả lời trực tiếp, chặn thông tin ngoài phạm vi câu hỏi |

## 6. Nhận Xét Về Nhóm Adversarial

Điểm trung bình của nhóm adversarial là 0.409, thấp hơn factual 0.452 nhưng cao hơn multi_hop 0.390. Điều này cho thấy các câu hỏi bẫy về phiên bản/chính sách vẫn khó, nhưng điểm yếu lớn nhất của pipeline hiện tại nằm ở câu hỏi nhiều bước và câu hỏi cần tính toán. Có 2 câu adversarial nằm trong bottom 10: câu về chu kỳ đổi mật khẩu và câu về dùng VPN cá nhân. Cả hai đều cần retrieval đúng tài liệu chính sách hiện hành, không chỉ lấy đoạn bảo mật gần nghĩa.
