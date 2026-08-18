# Failure Analysis — Lab 18: Production RAG

**Sinh viên:** Nguyễn Hoàng Vũ · **MSSV:** 2A202601941
**Bài tập cá nhân — implement toàn bộ M1 → M5**
**Ngày chạy:** 18/08/2026 · **Test set:** 20 câu hỏi

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8250 | 0.8083 | **−0.0167** |
| Answer Relevancy | 0.7216 | 0.7799 | **+0.0584** |
| Context Precision | 0.9250 | 0.9417 | **+0.0167** |
| Context Recall | 0.9250 | 0.8500 | **−0.0750** |

**Cấu hình so sánh**

| | Naive Baseline | Production |
|---|---|---|
| Chunking | `chunk_basic` — theo paragraph, ~410 ký tự | `chunk_hierarchical` — parent 2048 / child 256 |
| Enrichment | không | combined 1-call (context + summary + HyQA + metadata) |
| Search | dense-only (bge-m3) | BM25 (underthesea) + dense + RRF |
| Reranking | không | bge-reranker-v2-m3, top-20 → top-3 |
| Số chunks | 51 | 108 |

### Nhận xét quan trọng: production KHÔNG thắng toàn diện

Hai metric tăng, hai metric giảm. Đây không phải lỗi implement mà là **đánh đổi có thể giải thích được**:

- **`context_precision` +0.0167 và `answer_relevancy` +0.0584** — đúng như kỳ vọng. Reranking lọc nhiễu ra khỏi top-3, enrichment thêm câu mô tả ngữ cảnh giúp câu trả lời bám sát câu hỏi hơn.
- **`context_recall` −0.0750** — nguyên nhân là kích thước chunk. Baseline lấy 3 chunk × ~410 ký tự ≈ **1230 ký tự** context. Production lấy 3 chunk × 256 ký tự ≈ **768 ký tự**. Chunk nhỏ giúp retrieve chính xác hơn nhưng mỗi chunk mang ít thông tin hơn, nên khi đáp án trải dài qua nhiều câu thì bị cắt mất một phần.
- **`faithfulness` −0.0167** — kéo xuống bởi các câu yêu cầu tính toán số học (xem #2 và #5), nơi LLM tự suy diễn thay vì bám context.

**Kết luận:** với corpus nhỏ và câu hỏi lookup đơn giản, chunk lớn của baseline đã đủ tốt. Production chỉ thực sự thắng khi corpus lớn hơn và nhiễu nhiều hơn — nhưng nó **để lộ ra lớp lỗi mới** ở tầng generation mà baseline che giấu.

---

## Phân bố lỗi trên 10 câu tệ nhất

| Metric tệ nhất | Số câu | Ý nghĩa |
|---|---|---|
| `faithfulness` | 5 | LLM bịa hoặc từ chối trả lời — **lỗi tầng generation** |
| `context_recall` | 4 | Retrieval bỏ sót một phần đáp án — **lỗi tầng retrieval** |
| `answer_relevancy` | 1 | Trả lời đúng nhưng thiếu trọng tâm |

Một nửa số lỗi nằm ở **generation**, không phải retrieval. Đây là điều bất ngờ nhất: mọi công sức tối ưu M1–M3 đều nhắm vào retrieval, trong khi `context_precision` đã đạt 0.9417 — nút thắt thật sự đã dịch sang chỗ khác.

---

## Bottom-5 Failures

### #1 — Xung đột phiên bản (avg 0.396, tệ nhất)

- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải được thay đổi mỗi 120 ngày. Chính sách cũ yêu cầu 90 ngày nhưng đã bị thay thế.
- **Got:** `Không tìm thấy.`
- **Scores:** faithfulness **0.0** · answer_relevancy **0.0** · context_precision 0.5833 · context_recall **1.0**
- **Worst metric:** faithfulness

**Context thực tế đã đưa vào LLM:**

```
[0] "...Mật khẩu phải được thay đổi **mỗi 90 ngày**..."     ← mat_khau_v1 (hết hiệu lực)
[1] "...Mật khẩu phải được thay đổi **mỗi 120 ngày**..."    ← mat_khau_v2 (hiện hành)
[2] "...Văn bản này thay thế Chính sách mật khẩu v1.0..."
```

**Error Tree:**

```
Output sai ("Không tìm thấy")
 └─ Context có chứa đáp án? → CÓ (context_recall = 1.0)
     └─ Context có sạch không? → KHÔNG (precision 0.5833)
         └─ Chứa 2 đáp án MÂU THUẪN TRỰC TIẾP: 90 ngày vs 120 ngày
             └─ LLM phân biệt được cái nào hiện hành? → KHÔNG
                 └─ Prompt yêu cầu "chỉ trả lời dựa trên context"
                     → LLM gặp mâu thuẫn, chọn cách an toàn: từ chối trả lời
```

**Root cause:** Đây **không phải lỗi retrieval** — retrieval đã làm đúng việc của nó (recall 1.0). Lỗi nằm ở chỗ hệ thống không có cơ chế nào cho LLM biết văn bản nào còn hiệu lực.

Nghiêm trọng hơn: **M5 enrichment đã làm vấn đề tệ đi**. Câu context sinh ra cho cả hai chunk gần như giống hệt nhau:

```
chunk v1 → "Đoạn văn nằm trong phần hướng dẫn về bảo mật mật khẩu trong tài liệu."
chunk v2 → "Đoạn văn nằm trong phần hướng dẫn về bảo mật mật khẩu trong tài liệu."
```

Enrichment đã **xoá sạch dấu vết phiên bản**. Thông tin `source: mat_khau_v1.md` / `mat_khau_v2.md` vẫn nằm trong metadata nhưng `pipeline.run_query()` chỉ ghép `r.text` vào prompt — metadata không bao giờ đến được LLM.

Reranking cũng không cứu được: khi đo riêng ở M3, khoảng cách điểm giữa hai phiên bản chỉ **0.0025** (0.9895 vs 0.9870) — hai văn bản gần như đồng nhất về ngữ nghĩa nên cross-encoder không có cơ sở để phân biệt.

**Suggested fix (theo thứ tự ưu tiên):**

1. **Đưa metadata vào context string** — ghép `[Nguồn: {source}]` trước mỗi chunk trong prompt. Rẻ nhất, sửa 1 dòng trong `run_query()`.
2. **Lọc theo ngày hiệu lực** — parse `Ngày hiệu lực` trong header, khi 2 chunk cùng chủ đề thì chỉ giữ bản mới nhất.
3. **Sửa prompt enrichment** — bắt buộc câu context phải nêu phiên bản và ngày hiệu lực.

---

### #2 — Suy luận số học nhiều bước (avg 0.637)

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn 15 ngày. Quá hạn 5 ngày, phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng, pro-rata ≈ **50.000 VNĐ** cho 5 ngày.
- **Got:** `15.000.000 × 2% = 300.000 VNĐ. Vậy nhân viên sẽ bị phạt 300.000 VNĐ.`
- **Scores:** faithfulness **0.0** · answer_relevancy 0.8807 · context_precision 1.0 · context_recall 0.6667

**Error Tree:**

```
Output sai (300.000 thay vì 50.000)
 └─ Context đúng? → CÓ, precision 1.0
     └─ Context đủ? → THIẾU (recall 0.6667) — không có quy tắc pro-rata
         └─ LLM làm gì khi thiếu thông tin? → TỰ SUY DIỄN
             └─ Bỏ qua bước chia tỷ lệ 5/30 ngày
                 → Ra đúng phí 1 tháng, sai đáp án bài toán
```

**Root cause:** Câu hỏi đòi 3 bước suy luận (xác định quá hạn → tính phí tháng → pro-rata theo ngày). Context có công thức 2%/tháng nhưng thiếu quy tắc pro-rata. LLM không nói "không đủ thông tin" mà tự điền vào khoảng trống — đúng định nghĩa hallucination, nên faithfulness = 0.

**Suggested fix:** Prompt phải yêu cầu nêu rõ từng bước tính và **dừng lại nếu thiếu dữ kiện**, thay vì tự suy diễn. Đây là lỗi prompt engineering, không phải lỗi retrieval.

---

### #3 — Câu hỏi phủ định (avg 0.750)

- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** **KHÔNG.** Nhân viên thử việc chưa được hưởng gói PVI. Chỉ được tham gia bảo hiểm xã hội bắt buộc.
- **Got:** `Nhân viên thử việc chưa được hưởng gói bảo hiểm sức khỏe PVI.`
- **Scores:** faithfulness **1.0** · answer_relevancy **0.0** · context_precision 1.0 · context_recall 1.0

**Error Tree:**

```
Retrieval hoàn hảo (3/4 metric = 1.0)
 └─ Nội dung trả lời đúng? → ĐÚNG
     └─ Vậy tại sao answer_relevancy = 0.0?
         └─ Câu hỏi dạng có/không nhưng câu trả lời KHÔNG mở đầu bằng "Không"
             └─ Thiếu luôn nửa sau: "chỉ được BHXH bắt buộc"
                 → RAGAS đánh giá là không trả lời trúng câu hỏi
```

**Root cause:** Đây là loại lỗi tinh vi nhất trong nhóm — **không có gì sai về mặt sự thật**. Với câu hỏi có/không, RAGAS đòi câu trả lời phải khẳng định trực tiếp. Câu trả lời hiện tại chỉ mô tả sự việc chứ không trả lời "có hay không".

**Suggested fix:** Thêm vào prompt: *"Nếu câu hỏi dạng có/không, bắt đầu bằng 'Có' hoặc 'Không', sau đó mới giải thích."* Sửa một dòng prompt, không đụng gì tới retrieval.

---

### #4 — Retrieval bỏ sót (avg 0.822)

- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16–30 ngày cần phê duyệt của **CEO**. Lưu ý: nghỉ trên 14 ngày, nhân viên **phải tự đóng phần bảo hiểm** của mình.
- **Got:** `Cần phê duyệt của Giám đốc điều hành (CEO).`
- **Scores:** faithfulness 1.0 · answer_relevancy 0.786 · context_precision 1.0 · **context_recall 0.5**

**Error Tree:**

```
Output đúng một nửa
 └─ Context đúng? → CÓ, precision 1.0
     └─ Context ĐỦ? → KHÔNG, recall 0.5 — mất vế bảo hiểm
         └─ Tại sao mất? → Đáp án nằm ở 2 section KHÁC NHAU của tài liệu
             └─ chunk_hierarchical cắt child = 256 ký tự
                 → 2 vế rơi vào 2 chunk khác nhau, chỉ 1 chunk lọt top-3
```

**Root cause:** Đây chính là mặt trái của `context_recall −0.075` đã nêu ở trên, thể hiện ở cấp độ một câu hỏi cụ thể. Child chunk 256 ký tự quá nhỏ để chứa trọn một quy định có nhiều vế.

**Suggested fix:** Dùng đúng cơ chế **parent-retrieval** mà `chunk_hierarchical` đã dựng sẵn — retrieve bằng child (chính xác) nhưng **trả về parent** (đủ ngữ cảnh). Hiện `pipeline.py` mới chỉ index child và trả về child; `parent_id` đã có trong metadata nhưng chưa được dùng. Đây là cải tiến đáng giá nhất và gần như miễn phí.

---

### #5 — Suy luận số học, thiếu lập luận (avg 0.826)

- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Cam kết làm việc tối thiểu 1 năm. Nghỉ sau 8 tháng là trước hạn → hoàn trả **100% = 25.000.000 VNĐ**.
- **Got:** `100% chi phí đào tạo đã được tài trợ, tức là 25 triệu VNĐ.`
- **Scores:** **faithfulness 0.5** · answer_relevancy 0.8018 · context_precision 1.0 · context_recall 1.0

**Error Tree:**

```
Con số cuối ĐÚNG nhưng faithfulness chỉ 0.5
 └─ Context đủ? → CÓ (recall 1.0, precision 1.0)
     └─ Đáp án đúng? → ĐÚNG (25 triệu)
         └─ Vậy sao mất điểm? → Trả lời BỎ QUA bước lập luận
             └─ Không nêu điều kiện "cam kết 1 năm" và "8 < 12 tháng"
                 → RAGAS không truy vết được kết luận về context → chấm 0.5
```

**Root cause:** LLM đưa ra kết luận đúng nhưng **giấu lập luận**. RAGAS `faithfulness` kiểm tra từng mệnh đề trong câu trả lời có truy được về context hay không; mệnh đề "phải hoàn 100%" bị treo lơ lửng vì thiếu tiền đề.

**Suggested fix:** Prompt yêu cầu nêu điều kiện áp dụng trước khi kết luận. Cùng một hướng sửa với #2.

---

## Case Study (cho presentation)

**Question chọn phân tích:** *"Bao lâu phải đổi mật khẩu một lần?"* (#1)

Chọn câu này vì nó là **phản ví dụ hoàn hảo** cho giả định "retrieval tốt thì RAG tốt": mọi chỉ số retrieval đều ổn, hệ thống vẫn trả lời sai hoàn toàn.

**Error Tree walkthrough:**

1. **Output đúng?** → KHÔNG. Trả `"Không tìm thấy."` trong khi đáp án là 120 ngày.
2. **Context đúng?** → CÓ. `context_recall = 1.0`, đáp án nằm ngay trong context[1].
3. **Query rewrite OK?** → CÓ. Cả BM25 lẫn dense đều tìm đúng nhóm tài liệu mật khẩu.
4. **Fix ở bước nào?** → **Không phải retrieval.** Fix ở tầng ghép context và tầng metadata: LLM cần biết chunk nào thuộc phiên bản nào.

**Bài học rút ra:** ba module M1–M3 đều tối ưu cho retrieval, nhưng lỗi tệ nhất của hệ thống lại nằm ngoài phạm vi cả ba. `context_precision = 0.9417` nhìn rất đẹp nhưng che mất chuyện 0.58 còn lại ở câu #1 chính là hai văn bản mâu thuẫn nhau.

**Nếu có thêm 1 giờ, sẽ optimize theo thứ tự:**

1. **Ghép `source` vào context string** (~5 phút) — sửa `run_query()` thành `f"[Nguồn: {meta['source']}]\n{text}"`. Chỉ riêng việc này có thể cứu câu #1.
2. **Bật parent-retrieval** (~15 phút) — `parent_id` đã sẵn trong metadata, chỉ cần map child → parent trước khi đưa vào prompt. Nhắm thẳng vào `context_recall −0.075` và câu #4.
3. **Siết prompt generation** (~10 phút) — bắt buộc nêu từng bước tính, trả lời có/không phải mở đầu bằng "Có"/"Không", thiếu dữ kiện thì phải nói thiếu. Nhắm vào #2, #3, #5 — tức **3 trong 5 lỗi tệ nhất**.
4. **Lọc phiên bản theo ngày hiệu lực** (~30 phút) — giải pháp triệt để cho lớp lỗi version.

Đáng chú ý: **không có mục nào trong danh sách này là thay model hay chỉnh tham số retrieval.** Sau khi retrieval đã đạt ~0.94 precision, lợi ích biên nằm hết ở tầng prompt và metadata.
