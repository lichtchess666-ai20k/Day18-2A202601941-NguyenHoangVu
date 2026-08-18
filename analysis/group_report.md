# Group Report — Lab 18: Production RAG

**Hình thức:** Bài tập **cá nhân** — một người implement toàn bộ 5 modules
**Sinh viên:** Nguyễn Hoàng Vũ · **MSSV:** 2A202601941
**Ngày:** 18/08/2026

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Nguyễn Hoàng Vũ | M1: Chunking | ☑ | 13/13 |
| Nguyễn Hoàng Vũ | M2: Hybrid Search | ☑ | 5/5 |
| Nguyễn Hoàng Vũ | M3: Reranking | ☑ | 5/5 |
| Nguyễn Hoàng Vũ | M4: Evaluation | ☑ | 4/4 |
| Nguyễn Hoàng Vũ | M5: Enrichment | ☑ | 10/10 |

**Tổng: 37/37 tests pass · 0 TODO còn lại trong `src/`**

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.8250 | 0.8083 | −0.0167 |
| Answer Relevancy | 0.7216 | 0.7799 | **+0.0584** |
| Context Precision | 0.9250 | 0.9417 | **+0.0167** |
| Context Recall | 0.9250 | 0.8500 | −0.0750 |

Cả 4 metric của production đều ≥ 0.75. Test set: 20 câu hỏi.

## Latency Breakdown

Đo trên CPU (không GPU), Windows 11, Python 3.11.4, torch 2.13.0+cpu.

### Pipeline build (một lần)

| Bước | Thời gian | Ghi chú |
|------|-----------|---------|
| M1 Chunking | ~0.0s | 108 child chunks từ 26 documents |
| M5 Enrichment | **301.4s** | 108 chunks × 1 API call ≈ 2.8s/chunk |
| M2 Indexing | 30.9s | encode bge-m3 + upsert Qdrant |
| M3 Load reranker | ~0.0s | đã cache ở module level |
| **Tổng build** | **~332s** | |

### Per-query (20 câu)

| Bước | Thời gian | Ghi chú |
|------|-----------|---------|
| BM25 search | < 50ms | in-memory, không đáng kể |
| Dense search | ~200ms | encode query + Qdrant query_points |
| **M3 Rerank top-20 → top-3** | **~9,083ms** | đo riêng: min 7,719 / max 10,681 (3 lần) |
| LLM generation | ~2,000s ⁽¹⁾ | gpt-4o-mini |

⁽¹⁾ ~2 giây, không phải 2000s.

### Evaluation

| Bước | Thời gian |
|------|-----------|
| RAGAS 4 metrics × 20 câu | 36.7s |
| **Tổng cả run (baseline + production)** | **611.6s** |

### Nhận xét về latency

**Reranking chiếm ~80% thời gian mỗi query.** `bge-reranker-v2-m3` là XLM-RoBERTa-large (568M tham số); mỗi query phải forward 20 cặp (query, doc) qua transformer và **không thể precompute** như embedding — đó là cái giá phải trả cho độ chính xác của cross-encoder.

Ba hướng giảm:

| Cách | Ước tính | Đánh đổi |
|---|---|---|
| Giảm `HYBRID_TOP_K` 20 → 10 | ~4.5s/query | Có thể mất recall |
| Chuyển sang `FlashrankReranker` | ~0.2s/query | Độ chính xác thấp hơn |
| Chạy GPU | ~0.5s/query | Cần phần cứng |

`FlashrankReranker` đã được implement sẵn trong `src/m3_rerank.py` để có thể đổi ngay khi cần.

## Key Findings

### 1. Biggest improvement: Answer Relevancy +0.0584

Đến từ hai nguồn cộng hưởng: reranking đẩy chunk đúng lên top-3 (precision 0.9250 → 0.9417), và enrichment prepend một câu mô tả ngữ cảnh giúp LLM hiểu chunk đang nói về chủ đề gì.

Minh chứng cụ thể đo được ở M3 với query *"Nhân viên được nghỉ phép bao nhiêu ngày một năm?"*:

| | Top-1 | Top-2 | Top-3 |
|---|---|---|---|
| Hybrid (trước rerank) | v2023 (hết hiệu lực) | nghỉ phép **không lương** | v2024 |
| Sau rerank | **v2024 (hiện hành)** | v2023 | v2024 |

Reranker loại hẳn hai document nhiễu và đưa đúng bản hiện hành lên đầu.

### 2. Biggest challenge: Context Recall giảm 0.0750

Production **thua** baseline ở metric này. Nguyên nhân thuần tuý là kích thước chunk: baseline đưa vào ~1230 ký tự context (3 × 410), production chỉ ~768 ký tự (3 × 256).

Bài học: `child_size = 256` tối ưu cho *độ chính xác của phép so khớp* nhưng lại phản tác dụng ở *lượng thông tin đưa cho LLM*. Cơ chế parent-retrieval mà `chunk_hierarchical()` đã dựng sẵn chính là lời giải — nhưng `pipeline.py` chưa dùng tới, dù `parent_id` đã nằm sẵn trong metadata.

### 3. Surprise finding: một nửa số lỗi nằm ở generation, không phải retrieval

Trong 10 câu tệ nhất: 5 câu worst metric là `faithfulness`, chỉ 4 câu là `context_recall`.

Toàn bộ M1–M3 đều tối ưu cho retrieval, nhưng sau khi `context_precision` đạt 0.9417 thì nút thắt đã dịch sang tầng prompt. Câu tệ nhất (*"Bao lâu phải đổi mật khẩu một lần?"*, avg 0.396) có `context_recall = 1.0` — retrieval hoàn hảo, hệ thống vẫn trả lời `"Không tìm thấy."` vì context chứa cả 90 ngày (v1) lẫn 120 ngày (v2) mà không có cách nào phân biệt bản nào còn hiệu lực.

Phát hiện phụ đáng lo hơn: **M5 enrichment đã làm câu này tệ đi**. Câu context sinh cho hai chunk v1/v2 giống hệt nhau và không nhắc gì tới phiên bản, tức là enrichment đã xoá mất đúng thông tin cần để phân biệt.

## Presentation Notes (5 phút)

**1. RAGAS scores (naive vs production)**
Cả 4 metric production ≥ 0.75. Nhưng chỉ 2/4 metric tốt hơn baseline — nhấn mạnh rằng "production" không đồng nghĩa "tốt hơn ở mọi chỉ số".

**2. Biggest win — module nào, tại sao**
M3 Reranking. Có bằng chứng trước/sau rõ ràng: loại 2 document nhiễu khỏi top-3 và sửa được thứ tự phiên bản. Nhưng phải nói kèm cái giá: ~9 giây/query trên CPU, chiếm 80% thời gian mỗi truy vấn.

**3. Case study — Error Tree walkthrough**
Câu *"Bao lâu phải đổi mật khẩu một lần?"*:
Output sai → Context có đáp án? CÓ (recall 1.0) → Context sạch? KHÔNG (precision 0.58) → Chứa 2 đáp án mâu thuẫn 90 vs 120 ngày → LLM không phân biệt được bản hiện hành → từ chối trả lời.
**Fix không nằm ở retrieval** mà ở tầng ghép context: đưa `source`/phiên bản vào prompt.

**4. Next optimization nếu có thêm 1 giờ**

| Ưu tiên | Việc | Thời gian | Nhắm vào |
|---|---|---|---|
| 1 | Ghép `[Nguồn: {source}]` vào context string | ~5 phút | Câu #1 (xung đột phiên bản) |
| 2 | Bật parent-retrieval (dùng `parent_id` sẵn có) | ~15 phút | `context_recall`, câu #4 |
| 3 | Siết prompt: nêu từng bước tính, trả lời có/không trực tiếp | ~10 phút | Câu #2, #3, #5 |
| 4 | Lọc phiên bản theo ngày hiệu lực | ~30 phút | Toàn bộ lớp lỗi version |

Không có mục nào là đổi model hay chỉnh tham số retrieval — sau khi precision đạt ~0.94, lợi ích biên nằm hết ở tầng prompt và metadata.
