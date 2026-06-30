# Báo Cáo Bias Của LLM Judge - Phase B

**Sinh viên:** Vũ Văn Huy  
**Mã học viên:** 2A202600750  
**Ngày:** 2026-06-30  
**Judge model:** mimo-v2.5-pro, mặc định dùng deterministic fallback

## 1. Kết Quả Pairwise Judge

| # | Tóm tắt câu hỏi | Câu thắng | Lý do tóm tắt |
|---:|---|---|---|
| 1 | Số ngày nghỉ phép năm | A | Câu A cụ thể hơn và nhắc đúng chính sách hiện hành năm 2024 |

## 2. Kết Quả Swap-and-Average

| # | Winner lần 1 | Winner lần 2 sau khi swap | Kết quả cuối | Nhất quán vị trí? |
|---:|---|---|---|---|
| 1 | A | A | A | Có |

**Tỷ lệ position bias:** 0.0% (= 0 trường hợp không nhất quán / 1 trường hợp)

## 3. Phân Tích Cohen's Kappa

Báo cáo sinh ra sử dụng 10 nhãn human có sẵn trong `human_labels_10q.json` để kiểm tra đường chạy calibration. Judge fallback được ánh xạ về cùng dạng nhãn nhị phân với human labels.

| Chỉ số | Giá trị |
|---|---:|
| Số nhãn human | 10 |
| Số nhãn judge | 10 |
| Cohen's kappa | 1.000 |
| Diễn giải | almost perfect |

Trong môi trường production, chỉ số này nên được tính lại trên output judge thật cho cùng 10 ví dụ đã gán nhãn. Giá trị hiện tại chủ yếu xác nhận rằng phần tính kappa và cấu trúc báo cáo hoạt động đúng.

## 4. Verbosity Bias

Trong các trường hợp có winner rõ ràng:

| Chỉ số | Giá trị |
|---|---:|
| A thắng và A dài hơn B | 1 |
| B thắng và B dài hơn A | 0 |
| Tổng số trường hợp có winner rõ ràng | 1 |
| Tỷ lệ verbosity bias | 100.0% |

Vì mẫu hiện tại chỉ có một cặp so sánh rõ ràng, tỷ lệ verbosity bias chưa có ý nghĩa thống kê mạnh. Tuy nhiên, đây vẫn là tín hiệu cần theo dõi: câu trả lời dài hơn có thể thắng nếu đồng thời chứa thêm bằng chứng về phiên bản/chính sách mới.

## 5. Nhận Xét Chung

Cơ chế swap-and-average đã được bật và cho kết quả nhất quán trên cặp demo. Position bias thấp trong mẫu hiện tại, nhưng mẫu được giữ nhỏ để chạy lab nhanh, ổn định và tiết kiệm chi phí API. Khi chạy production hoặc đánh giá nghiêm túc hơn, nên bật `LAB24_USE_LLM_JUDGE=1`, chạy ít nhất 30-50 cặp câu trả lời, và giữ deterministic fallback cho CI smoke test.
