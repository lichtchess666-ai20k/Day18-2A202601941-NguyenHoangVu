# Individual Reflection — Lab 18: Production RAG

**Tên:** Nguyễn Hoàng Vũ · **MSSV:** 2A202601941
**Module phụ trách:** Toàn bộ M1 → M5 (bài tập cá nhân)
**Ngày:** 18/08/2026

---

## Phần 1: Mapping bài giảng → code

| Lecture Concept | Module | Hàm cụ thể | Observation (số liệu đo được) |
|----------------|--------|-------------|-------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 tạo **208 chunks** vs basic **51 chunks** — tệ hơn baseline. Đo phân bố cosine giữa các câu liên tiếp: p50 = 0.613, p90 = 0.715, **0.0% số cặp đạt ≥ 0.85**. Ngưỡng 0.85 hợp lý cho tiếng Anh nhưng `all-MiniLM-L6-v2` nén similarity tiếng Việt xuống dải 0.5–0.7 → mọi câu bị tách riêng. Ngưỡng ~0.6 mới thực sự nhóm câu (96 chunks). |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Parent 2048 / child 256 → 11 parents, 104 children (avg 200 ký tự). Cơ chế parent-retrieval đã dựng sẵn `parent_id` nhưng `pipeline.py` chưa dùng — đây là nguyên nhân `context_recall` tụt 0.075. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF giải quyết đúng điểm yếu của từng bên. Với query "nghỉ phép bao nhiêu ngày": **BM25 trượt top-1** (chọn "nghỉ phép không lương" vì trùng nhiều từ khoá), **dense đúng** nhưng bỏ sót; RRF giữ được doc đúng ở #1 và cứu v2024 vào top-3. |
| Vietnamese segmentation | M2 | `segment_vietnamese()` | underthesea nối từ ghép bằng `_` ("nghỉ_phép"). Nếu không `replace("_", " ")` thì query 2 token không bao giờ khớp document 1 token → BM25 chết im lặng, không báo lỗi. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Latency **~9,083ms/query** (20 docs, CPU) — chiếm ~80% thời gian mỗi truy vấn. Precision cải thiện thật: loại 2 doc nhiễu khỏi top-3 và đưa v2024 (hiện hành) vượt v2023. Nhưng khoảng cách chỉ **0.0025** → không đáng tin cho việc phân biệt phiên bản. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Metric thấp nhất là `answer_relevancy` (0.7799). Bất ngờ: `context_recall` **giảm** 0.075 so với baseline vì chunk nhỏ hơn (768 vs 1230 ký tự context). 5/10 lỗi tệ nhất có worst metric là `faithfulness` → nút thắt nằm ở generation chứ không phải retrieval. |
| Contextual embeddings | M5 | `contextual_prepend()` / `_enrich_single_call()` | Combined mode: 1 API call/chunk thay vì 4, 2.8s/chunk, 301s cho 108 chunks. **Phản tác dụng ngoài dự kiến:** câu context sinh cho `mat_khau_v1` và `mat_khau_v2` giống hệt nhau, xoá mất dấu vết phiên bản — góp phần trực tiếp gây ra lỗi tệ nhất của cả hệ thống. |

---

## Phần 2: Khó khăn & cách giải quyết

### 2.1. `ValueError` từ numpy chỉ lộ ra ở dữ liệu thật (khó nhất)

**Exact error:**
```
⚠️  RAGAS evaluation failed: The truth value of an array with more than one element
is ambiguous. Use a.any() or a.all()
```

**Bối cảnh:** Toàn bộ 37 test pass, smoke test RAGAS với 2 câu hỏi cho điểm đẹp (faithfulness 1.0). Nhưng khi chạy `main.py` thật thì cả baseline lẫn production đều trả về **0.0000 cho cả 4 metric**. Mất 10 phút chạy mới biết.

**Cách debug:** Vì đã bọc `try/except` nên pipeline không crash mà âm thầm trả zeros — điều này vừa cứu vừa hại. Truy ngược vào code của mình, thủ phạm là:

```python
contexts=list(row.get("contexts", []) or []),
```

Toán tử `or` gọi `bool()` lên giá trị bên trái. RAGAS trả `contexts` dưới dạng **numpy array**, và `bool()` của array nhiều phần tử là ambiguous → ValueError.

**Tại sao smoke test không bắt được:** smoke test dùng mỗi câu **1 context** → `bool(array 1 phần tử)` hợp lệ, chạy ngon. Pipeline thật trả **3 context** (`RERANK_TOP_K = 3`) → vỡ. Đây là bug chỉ xuất hiện khi `len(contexts) > 1`.

**Fix:**
```python
def _as_list(value) -> list:
    if value is None:
        return []
    return [str(item) for item in value]
```

**Bài học:** không bao giờ dùng `or` để đặt giá trị mặc định cho thứ có thể là numpy array/pandas Series. Và quan trọng hơn: **test smoke phải tái hiện đúng hình dạng dữ liệu thật**, không phải chỉ đúng kiểu dữ liệu. Sau khi sửa, tôi kiểm chứng lại bằng đúng điều kiện gây lỗi (3 context/câu) trước khi chạy lại 10 phút.

**Thời gian debug:** ~15 phút, cộng 10 phút chạy lại pipeline.

### 2.2. `os.rename()` crash trên Windows ở lần chạy thứ hai

**Exact error:**
```
FileExistsError: [WinError 183] Cannot create a file when that file already exists:
'ragas_report.json' -> 'reports/ragas_report.json'
```

**Bối cảnh:** Đây là bug trong `main.py` của đề bài, không phải code tôi viết. Lần chạy đầu tiên bình thường; lần thứ hai crash sau khi đã in xong toàn bộ điểm số.

**Nguyên nhân:** `os.rename()` trên POSIX ghi đè im lặng, nhưng trên Windows thì **ném lỗi nếu file đích đã tồn tại**. Vì lần chạy 1 đã tạo `reports/ragas_report.json`, lần 2 không move được.

**Hậu quả nguy hiểm:** report tốt nằm ở thư mục gốc, còn `reports/` vẫn giữ bản cũ toàn số 0. Nếu không kiểm tra, tôi đã nộp nhầm file zeros.

**Fix:** đổi sang `os.replace()` — ghi đè nguyên tử trên mọi hệ điều hành.

**Bài học:** code chạy được trên máy giảng viên (macOS/Linux) không đảm bảo chạy trên Windows. Và exit code 0 không có nghĩa là thành công — ở đây pipe qua `grep` đã che mất exit code thật của Python.

### 2.3. Console Windows không in được emoji

**Exact error:**
```
UnicodeEncodeError: 'charmap' codec can't encode characters in position 2-3:
character maps to <undefined>
  File "...\encodings\cp1258.py", line 19, in encode
```

**Nguyên nhân:** Console dùng codepage **cp1258** (tiếng Việt legacy), không encode nổi emoji `⚠️` trong `load_documents()`. Toàn bộ scaffold dùng emoji nên `pipeline.py` và `main.py` đều chết giữa chừng.

**Fix:** `PYTHONIOENCODING=utf-8`. Không phải sửa code.

### 2.4. `pip install` hỏng vì hai tiến trình chạy song song

**Exact error:**
```
TypeError: expected string or bytes-like object, got 'NoneType'
  File "...\pip\_internal\metadata\importlib\_dists.py", line 178, in version
    return parse_version(self._dist.version)
```

**Cách debug:** Ban đầu tưởng xung đột version, nhưng `pip install --dry-run` lại resolve sạch. Nghi ngờ nảy sinh khi kiểm tra `site-packages` hai lần cách nhau vài phút thì **số gói tự tăng thêm 6** dù tôi không cài gì. Kết luận: có tiến trình pip thứ hai đang ghi vào cùng thư mục, pip 23.1.2 đọc trúng metadata đang ghi dở nên vỡ.

**Fix:** đảm bảo chỉ một tiến trình pip, nâng pip 23.1.2 → 26.2.1.

### 2.5. Kiến thức còn thiếu và cách bổ sung

| Thiếu | Cách bổ sung |
|---|---|
| Không biết ngưỡng similarity phụ thuộc mạnh vào ngôn ngữ của model | Đo phân bố thực tế (percentile) trước khi chọn threshold, thay vì tin giá trị mặc định |
| Chưa nắm cross-encoder đắt hơn bi-encoder bao nhiêu | Đã benchmark: ~9s cho 20 cặp trên CPU. Lần sau phải tính latency budget **trước** khi chọn kiến trúc |
| Chưa biết RAGAS `faithfulness` chấm theo từng mệnh đề | Đọc lại metric definition — giải thích được vì sao câu #5 có đáp án đúng nhưng chỉ được 0.5 |

---

## Phần 3: Action Plan cho project

> ⚠️ **Phần này cần bạn tự điền theo project thực tế của mình.** Khung dưới đây dựa trên các kết luận đo được từ lab, điền tên project và điều chỉnh timeline cho phù hợp.

#### Plan áp dụng (dựa trên kết quả đo được ở lab)

1. **[ ] Chunking strategy: hierarchical (parent 2048 / child 256) + parent-retrieval**
   Lý do: lab cho thấy child nhỏ giúp precision (0.9250 → 0.9417) nhưng làm tụt recall (0.9250 → 0.8500). Parent-retrieval lấy được cả hai. **Không dùng semantic chunking** trừ khi đã đo phân bố similarity trên đúng corpus và ngôn ngữ của mình.

2. **[ ] Search: Hybrid BM25 + Dense + RRF**
   Lý do: đo được BM25 trượt top-1 ở query "nghỉ phép" (trùng từ khoá nhưng sai ý), dense thì bỏ sót. RRF sửa được cả hai. Tiếng Việt bắt buộc segment bằng underthesea và nhớ `replace("_", " ")`.

3. **[ ] Reranking: CÓ, nhưng cân nhắc theo latency budget**
   `bge-reranker-v2-m3` chính xác nhất nhưng ~9s/query trên CPU. Nếu project cần realtime → dùng `flashrank` hoặc chạy GPU. Nếu batch/offline → giữ bge-reranker.

4. **[ ] Evaluation: RAGAS 4 metrics + failure analysis theo Diagnostic Tree**
   Quan trọng: **luôn chạy baseline trước**. Nếu lab này không có baseline, tôi đã tưởng production tốt trong khi thực tế nó thua ở 2/4 metric.

5. **[ ] Enrichment: contextual prepend — nhưng PHẢI giữ metadata phân biệt**
   Bài học đắt nhất của lab: enrichment xoá mất dấu vết phiên bản và trực tiếp gây ra lỗi tệ nhất. Prompt enrichment **bắt buộc** phải nêu nguồn/phiên bản/ngày hiệu lực.

6. **[ ] Ưu tiên tầng generation ngang với retrieval**
   5/10 lỗi tệ nhất là `faithfulness`. Prompt phải: nêu từng bước tính toán, trả lời có/không trực tiếp, và **nói rõ khi thiếu dữ kiện** thay vì tự suy diễn.

#### Timeline

- **Tuần `[X]`:** `[dựng baseline + bộ test set có ground truth, chạy RAGAS lấy số gốc]`
- **Tuần `[Y]`:** `[áp dụng hybrid search + hierarchical chunking, đo lại]`
- **Tuần `[Z]`:** `[thêm reranking nếu latency cho phép, siết prompt, phân tích failure]`

---

## Tự đánh giá

> ⚠️ Phần tự chấm bên dưới cần bạn tự điền theo cảm nhận thật của mình.

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | `[  ]` |
| Code quality | `[  ]` |
| Teamwork | N/A — bài cá nhân |
| Problem solving | `[  ]` |

### Nếu làm lại

- **Sẽ làm khác:** chạy smoke test với đúng hình dạng dữ liệu thật (3 context, không phải 1) **trước khi** chạy pipeline 10 phút. Bug numpy `or` lẽ ra bị bắt ngay từ đầu.
- **Module muốn thử tiếp:** parent-retrieval trong M1 — `parent_id` đã có sẵn nhưng chưa dùng, và đây là cách rẻ nhất để lấy lại 0.075 `context_recall` đã mất.
