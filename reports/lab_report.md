# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Nguyễn Văn Minh
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 19/08/2026

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)
> **Tình huống thực tế:** Nêu ít nhất 1 tình huống cụ thể trong dữ liệu HackerNoon mà cơ chế Coreference Resolution phân giải sai hoặc gặp khó khăn. Hậu quả của nó đối với Knowledge Graph là gì?

*Trả lời:*
- **Cơ chế:** `resolve_coref_batch()` dùng `COREF_SYSTEM` với nguyên tắc conservative — chỉ phân giải đại từ (`it`, `they`, `the company`...) khi antecedent xuất hiện rõ ràng **trong cùng một chunk**; nếu mơ hồ thì giữ nguyên văn bản gốc và model tự ghi `unresolved_mentions`.
- **Tình huống khó khăn quan sát được:** Dữ liệu HackerNoon gồm nhiều bài tin tức ngắn (220 từ/chunk) viết về các thương vụ M&A liên tiếp nhau (ví dụ chuỗi tin Aeris–Ericsson, ServiceNow–NVIDIA/Cohere). Khi một chunk mở đầu bằng đại từ mà antecedent nằm ở chunk trước (do bị cắt bởi rolling-window chunking, `CHUNK_WORDS=220`), quy tắc conservative đúng theo thiết kế sẽ **từ chối phân giải** (giữ nguyên `it`/`the company`) thay vì suy diễn — đây là hành vi an toàn nhưng đổi lại làm giảm recall của bước NER+RE ngay sau đó, vì `extract_batch()` không biết `it` ám chỉ thực thể nào.
- **Hậu quả đối với Graph:** Không phải False Edge (vì cơ chế conservative ngăn được điều đó), mà là **Missing Edge** — quan hệ đúng ra tồn tại nhưng không được trích xuất vì entity chủ ngữ bị mơ hồ hóa thành đại từ không xác định. Đây chính là một phần lý do khiến GraphRAG bỏ sót thông tin so với Flat RAG trong benchmark thực tế (mục 4), vì Flat RAG vẫn giữ nguyên toàn văn chunk (kể cả đại từ mơ hồ) trong context, còn GraphRAG chỉ có thể truy xuất qua cạnh đã được trích xuất thành công.

---

### 2. Entity Resolution Threshold & Lexical Guard
> **Ngưỡng & Cơ chế Guard:** Bạn chọn ngưỡng cosine similarity là bao nhiêu cho vector matching? Trích dẫn 1 cặp thực thể có độ tương đồng vector cao ($> 0.85$) nhưng bị Lexical Guard chặn không cho gộp (Reject) và giải thích lý do.

*Trả lời:*
- **Ngưỡng cosine similarity:** `threshold = 0.90` cho vector ANN candidate search (`build_resolution_map()`, embedding `sentence-transformers/all-MiniLM-L6-v2`, FAISS `IndexFlatIP`). Sau khi qua ngưỡng vector, `merge_guard()` áp thêm Lexical Guard: cắt hậu tố công ty (`CORP_SUFFIXES`: inc/corp/ltd/llc...) rồi so khớp bằng `SequenceMatcher.ratio() >= 0.72`.
- **Cặp thực thể bị Guard chặn:** *(sẽ điền cặp cụ thể trích từ `entity_resolution_audit_df` — bảng audit đang được tái tạo lại vì bị mất khỏi RAM kernel giữa phiên làm bài; xem ghi chú cuối mục này).* Theo thiết kế guard, các cặp điển hình có nguy cơ bị `REJECT_GUARD` dù cosine cao là dạng **công ty mẹ vs sản phẩm cùng tên** (`Apple` vs `Apple Watch`) hoặc **người trùng họ** (`Sam Altman` vs `Steve Altman`) — cả hai đều được nêu là failure mode cần chặn trong ASSIGNMENT.md.
- **Lý do chặn:** `SequenceMatcher` đo độ tương đồng chuỗi ký tự (lexical), trong khi bước trước đó (`cosine similarity`) đo tương đồng *ngữ nghĩa* qua embedding. Hai đại lượng này không tương quan tuyệt đối: "Apple" và "Apple Watch" có cosine cao vì cùng ngữ cảnh (bài viết công nghệ, cùng nhắc tới hệ sinh thái Apple) nhưng là hai **thực thể khác loại** (Company vs Technology/Product) về bản chất — nếu gộp sẽ merge nhầm doanh thu/sự kiện của công ty mẹ vào một sản phẩm cụ thể, gây False Merge nghiêm trọng cho các câu hỏi factoid/financial.

> ⏳ **Ghi chú cập nhật:** `entity_resolution_audit_df` bị mất khỏi RAM kernel giữa phiên làm việc (không được persist ra đĩa ở lần chạy M2/M3 trước) trong khi báo cáo này được soạn. Pipeline trích xuất (Coreference → NER+RE → Entity Resolution) đang được chạy lại ở nền để tái tạo bảng audit thật; mục này sẽ được cập nhật với cặp thực thể cụ thể + `similarity` số thật ngay khi hoàn tất.

---

### 3. Đồ thị & Super-node Mitigation
> **Đặc trưng đồ thị & Cắt tỉa cạnh:** Top 3 thực thể có bậc (degree) cao nhất trong đồ thị là gì? Việc ưu tiên lấy $N$ cạnh ($N=50$) có `published_date` mới nhất tại các Super-node mang lại ưu điểm gì và có rủi ro tiềm ẩn nào?

*Trả lời:*
- **Top 3 Super-nodes:**

| Hạng | Tên thực thể | Loại thực thể (Type) | Bậc kết nối (Degree) |
|------|--------------|---------------------|----------------------|
| 1 | *(đang chờ `test_supernode_policy()`)* | | |
| 2 | | | |
| 3 | | | |

> ⏳ Bảng trên đang được điền từ Cypher `MATCH (n)-[r]-() RETURN n.name, count(r) ORDER BY count(r) DESC LIMIT 3` chạy trong `test_supernode_policy()` (cell 5.1). Ghi nhận nhanh: trên toàn bộ 50 câu Golden benchmark, cột `graph_supernode_events` trong `graphrag_eval_results.csv` bằng **0 cho mọi câu** — nghĩa là với quy mô đồ thị hiện tại (`EXTRACTION_MAX_CHUNKS=400`), **không có node nào thực sự vượt ngưỡng `degree > 100`** để kích hoạt cơ chế cắt tỉa trong lúc trả lời Golden Dataset. Cơ chế Super-node Mitigation vẫn hoạt động đúng (đã unit-test qua `test_supernode_policy()`) nhưng chưa bị "kiểm thử áp lực" thật ở quy mô dữ liệu lab hiện tại.

- **Ưu điểm & Rủi ro của Temporal Mitigation (ưu tiên 50 cạnh mới nhất theo `published_date`):**
  - *Ưu điểm:* Giữ `MAX_GRAPH_CONTEXT_CHARS`/`GLOBAL_EDGE_CAP` trong tầm kiểm soát, tránh bùng nổ token khi traverse qua các hub lớn (Microsoft, Google...); ưu tiên tin tức mới nhất phù hợp với bản chất "breaking news" của domain tech-news, nơi trạng thái sự kiện (deal đang đàm phán → đã hoàn tất) thay đổi theo thời gian.
  - *Rủi ro:* Cắt mất cạnh lịch sử quan trọng cho câu hỏi cross-doc/multi-hop cần **đối chiếu theo dòng thời gian** (ví dụ: "So sánh phát biểu đầu năm và cuối năm của công ty X") — nếu cả cạnh cũ lẫn mới đều liên quan tới câu trả lời, chính sách "chỉ lấy 50 cạnh mới nhất" có thể loại bỏ chính cạnh cũ mà câu hỏi cần, gây thiếu chứng cứ (giảm Comprehensiveness) mà không có cảnh báo tường minh cho người dùng.

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

Dữ liệu dưới đây được tính trực tiếp từ `outputs/graphrag_eval_results.csv` (50/50 câu Golden Dataset, đủ 3 nhóm factoid/multi-hop/cross-doc, LLM-as-a-Judge chấm thang 1–5).

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge, trung bình toàn bộ 50 câu):

| Tiêu chí đánh giá | Flat RAG | GraphRAG | Độ chênh lệch ($\Delta$ = Graph − Flat) | Nhận xét phân tích |
|-------------------|----------|----------|--------------------------|-------------------|
| **Comprehensiveness (1–5)** | 3.52 | 3.16 | −0.36 | GraphRAG kém hơn — chủ yếu do Missing Edge từ Coreference/Extraction (mục 1) và `EXTRACTION_MAX_CHUNKS=400` giới hạn phạm vi tri thức của Graph nhỏ hơn nhiều so với Flat RAG (vector index phủ toàn bộ `chunks_df`). |
| **Faithfulness (1–5)** | 3.56 | 3.28 | −0.28 | Tương tự — khi Graph traversal đi sai hướng (seed/entity resolution nhầm), câu trả lời trích dẫn sai nguồn dù vẫn có `chunk_id` kèm theo (xem ca lỗi G5000-26 bên dưới). |
| **Multi-hop Reasoning (1–5)** | 3.56 | 3.14 | −0.42 | Ngược kỳ vọng lý thuyết (GraphRAG được thiết kế để mạnh hơn ở multi-hop) — nguyên nhân gốc rễ là đồ thị tri thức quá thưa (chỉ 400/6000+ chunk được đưa qua NER+RE) nên nhiều "hop" cần thiết không tồn tại thành cạnh, buộc BFS traversal dừng sớm hoặc đi lạc. |
| **Latency trung bình (s)** | 7.17 | 6.90 | −0.27 | GraphRAG nhanh hơn nhẹ trong sample này — context graph (đã linearize) thường ngắn gọn hơn context vector top-k=6 chunk thô. |
| **Token usage trung bình** | 1031.1 | 908.3 | −122.8 | GraphRAG tiết kiệm token hơn ~12%, nhất quán với latency thấp hơn — đánh đổi Comprehensiveness lấy Cost/Latency. |

**Nhận xét tổng quát:** Ở quy mô dữ liệu giới hạn của lab (`Scale Guard`), GraphRAG rẻ và nhanh hơn Flat RAG nhưng **chất lượng câu trả lời (theo LLM Judge) lại thấp hơn** trên cả 3 tiêu chí — trái với kỳ vọng lý thuyết. Nguyên nhân gốc rễ không nằm ở kiến trúc retrieval (BFS + super-node cap hoạt động đúng) mà ở **độ phủ của Knowledge Graph**: `EXTRACTION_MAX_CHUNKS=400` chỉ trích xuất triples từ một phần nhỏ của 5000+ chunk trong `chunks_df`, trong khi Flat RAG index toàn bộ. Đây là trade-off của Scale Guard trong giờ lab, không phải giới hạn cố hữu của kiến trúc GraphRAG (xem mục 5).

#### Phân tích 2 Ca lỗi Điển hình:

1. **Ca lỗi Flat RAG thất bại (GraphRAG thành công) — `G5000-18` (cross-doc):**
   - *Câu hỏi:* "Why should rows 1943, 2566, and 3284 be treated as duplicate/follow-up evidence for one Microsoft incident rather than three independent outages?"
   - *Tại sao Flat RAG thất bại?* Flat RAG trả lời thẳng "the supplied context does not contain any information about rows 1943, 2566, or 3284" (điểm 1/1/1) — vector top-k=6 không truy được đủ 3 chunk liên quan cùng lúc vì chúng nằm rải rác trong không gian embedding (viết bởi 3 nguồn/thời điểm khác nhau về cùng 1 sự kiện), Flat RAG không có cơ chế liên kết chủ động giữa các chunk.
   - *GraphRAG đã giải quyết như thế nào?* Đạt điểm 2/5/2 — nhờ cạnh đồ thị nối các mention Microsoft lại qua entity chung, graph traversal lấy được ít nhất 1 chunk chứng cứ đúng (Reuters, cùng outage) và trả lời có căn cứ thay vì từ chối hoàn toàn, dù vẫn thiếu chi tiết ngày tháng/cyberattack attribution so với reference.

2. **Ca lỗi GraphRAG thất bại nặng — `G5000-26` (multi-hop):**
   - *Câu hỏi:* "What external technology provider is named inside Amazon's July AI-service expansion, and what other new AI capability is mentioned alongside it?"
   - *Nguyên nhân:* GraphRAG trả lời **sai thực thể** — nêu "OpenAI" thay vì "Cohere" (điểm 1/1/1, so với Flat RAG đạt 5/5/5). Đây là lỗi **graph traversal đi lạc**: seed entity extraction hoặc BFS đã bám theo cạnh `Amazon –PARTNERED_WITH/USES→ OpenAI` (tồn tại thật trong đồ thị từ context khác) thay vì cạnh đúng liên quan tới Cohere trong chunk gốc của câu hỏi — hệ quả của Entity Resolution/Extraction thưa khiến graph traversal "nhảy" sang nhánh liên quan nhưng sai ngữ cảnh thời gian (không phải July, không phải đúng bài báo).
   - *Đề xuất khắc phục:* (a) Thêm ràng buộc temporal/provenance khi traversal — ưu tiên cạnh có `published_date` khớp phạm vi thời gian câu hỏi thay vì chỉ lấy cạnh gần nhất theo degree cap; (b) tăng `EXTRACTION_MAX_CHUNKS` hoặc extract theo lô có định hướng theo entity của Golden Dataset để đồ thị đủ dày quanh các case đánh giá; (c) thêm bước "seed verification" — kiểm tra seed entity match có nằm trong cùng chunk với ngữ cảnh câu hỏi (ví dụ theo tên công ty được hỏi) trước khi mở rộng BFS.

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent
> **Trade-offs, Agent Control & Scale 350MB**

*Trả lời:*
- **Đánh đổi Quality vs Cost vs Latency:** Theo benchmark thực tế (mục 4), GraphRAG rẻ hơn ~12% token và nhanh hơn nhẹ, nhưng chất lượng thấp hơn Flat RAG ở quy mô đồ thị nhỏ (400 chunk extract). Bài học rút ra: GraphRAG chỉ thắng Flat RAG về *quality* khi đồ thị đủ dày (extraction coverage đủ lớn) để BFS traversal thực sự tìm được đường nối đa-hop; nếu không, chi phí xây dựng Knowledge Graph (extraction + entity resolution + Neo4j ingestion) không được đền bù bằng chất lượng câu trả lời — một trade-off quan trọng cần cân nhắc khi quyết định có đáng đầu tư GraphRAG hay không cho một tập dữ liệu cụ thể.
- **Quyết định từ chối AI Coding Agent:** Khi pipeline bị gián đoạn nhiều lần giữa giờ làm bài (kernel/notebook crash do hết hạn mức API — xem mục 2 Phần 2), phương án "an toàn" thường thấy là `Restart & Run All` từ đầu. Tôi đã **từ chối** cách này vì nó buộc tính lại toàn bộ embedding, gọi lại LLM cho coreference/extraction đã thành công trước đó, và mất kết nối Neo4j đang có — cực kỳ tốn kém khi hạn mức API đã cạn phần lớn. Thay vào đó, chọn giải pháp attach trực tiếp vào kernel đang sống (giữ nguyên `nodes_df`, FAISS index, Neo4j driver trong RAM) và chỉ resume đúng những cell còn thiếu — tiết kiệm đáng kể lượt gọi API còn lại.
- **Giải pháp scale 350MB:** Ở quy mô ~100.000 bài báo, bottleneck đầu tiên là **NER+RE extraction qua LLM** (hiện tại `EXTRACTION_MAX_CHUNKS=400`, tuần tự theo batch — không thể scale tuyến tính). Giải pháp: (1) Async batch extraction với worker queue + nhiều API key luân phiên để tránh trần TPD/TPM của một tài khoản duy nhất (rủi ro đã gặp thực tế trong lab này — 3/6 model của một tài khoản Groq cạn hạn mức ngày chỉ sau ~150 lượt gọi); (2) Entity Resolution chuyển từ `IndexFlatIP` (so khớp toàn bộ, $O(N^2)$) sang HNSW/IVF index để scale sub-linear theo số lượng thực thể; (3) Neo4j bulk insert theo batch lớn hơn kèm `PERIODIC COMMIT`/`apoc.periodic.iterate` để tránh transaction quá lớn; (4) Community Partitioning (Bonus) để chia đồ thị theo cụm chủ đề, giảm phạm vi BFS traversal mỗi truy vấn.

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code
| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|--------------------------|------------------|------------------------|-----------------------------|
| **Conservative Coreference** | Module 1 | `resolve_coref_batch()`, `run_coref()` | Đúng nguyên tắc "chỉ resolve khi antecedent rõ ràng cùng chunk"; đánh đổi recall lấy precision (mục 1 Phần 1). |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` (dùng để lọc trong `run_extraction()`) | Lọc cứng schema ngay tại tầng parse, chặn được node/relation ngoài allowlist trước khi vào Neo4j — đúng tinh thần production schema validation. |
| **Bulk Cypher Ingestion** | Module 2 | `insert_nodes_unwind()`, `insert_edges_unwind()` (cell 2.3) | Dùng `UNWIND $rows AS row` theo batch thay vì `CREATE`/`MERGE` từng dòng — đúng yêu cầu hiệu năng của đề bài. |
| **Entity Resolution & Union-Find** | Module 3 | `build_resolution_map()`, class `UF` | Kết hợp Manual Aliases → Vector ANN (cosine ≥ 0.90) → Lexical Guard (`SequenceMatcher` ≥ 0.72) → Union-Find gán canonical ID; đủ 4 tầng như thiết kế đề bài. |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()`, `recent_edges()`, `node_degree()` | Giới hạn cạnh theo `published_date DESC LIMIT 50` khi degree vượt ngưỡng; trên Golden Dataset 50 câu, `graph_supernode_events=0` toàn bộ — cơ chế đúng nhưng chưa bị stress-test ở quy mô lab hiện tại. |
| **LLM-as-a-Judge Evaluation** | Module 5 | `judge_answer()`, `judge_json()`, `run_evaluation()` | Chấm 3 tiêu chí (Comprehensiveness/Faithfulness/Multi-hop) kèm rationale; có checkpoint/resume (`outputs/graphrag_eval_checkpoint.csv`) — quan trọng vì pipeline bị gián đoạn nhiều lần do rate-limit trong lab. |

---

### 2. Quá trình Debugging & Bài học
- **Lỗi kỹ thuật phức tạp nhất gặp phải:** Chuỗi lỗi liên hoàn khi hạn mức API (Groq TPD) cạn kiệt giữa giờ làm bài, phơi ra nhiều bug tiềm ẩn trong code nền (LLM wrapper) mà bình thường không lộ ra:
  1. `NameError: ALLOWED_NODE_TYPES` trong `extract_seeds()` — do biến chỉ được định nghĩa muộn (cell 2.1) trong khi `extract_seeds()` ở cell 3.2 dùng nó làm global; khi kernel chạy không đúng thứ tự tuần tự (nhiều lần restart), lỗi này âm thầm bị `except` bắt và trả seed rỗng thay vì crash — làm suy giảm chất lượng GraphRAG mà không có cảnh báo rõ ràng.
  2. Cơ chế fail-close kiểm tra lỗi 401 (`"401" in err_str`) dùng so khớp chuỗi con ngây thơ — khớp nhầm với số ngẫu nhiên trong message rate-limit (ví dụ `"Requested 2401"`), khiến lỗi 429 tạm thời bị hiểu nhầm thành lỗi xác thực và dừng hẳn (fail-close) một cách oan uổng.
  3. Khi thêm cơ chế đổi model dự phòng (model fallback chain) để chống rate-limit, phát hiện tiếp: JSON-mode của model reasoning (`qwen/qwen3.6-27b`) xì khối `<think>...</think>` ra trước JSON thật, phá parser; và việc validate JSON chỉ xảy ra ở tầng ngoài `groq_chat()` nên một model trả "200 OK nhưng nội dung rác" không kích hoạt được retry/đổi model.
- **Cách xử lý thành công:** Sửa từng lớp theo đúng root cause thay vì patch bề mặt: dời constant lên đầu file làm single source of truth; đổi so khớp "401" sang regex biên từ (`\b401\b`); gộp việc validate JSON vào ngay trong vòng lặp retry/đổi model; thêm cache "model đã cạn quota trong phiên" để không lãng phí lượt gọi thử lại vô ích; và quan trọng nhất — thêm checkpoint/resume cho cả bước Coreference và NER+RE extraction (theo đúng pattern đã có sẵn ở M5 Evaluation) để không mất tiến trình mỗi lần pipeline bị gián đoạn giữa chừng vì rate-limit.

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)
- **Tên đồ án / Dự án:** Trợ lý tra cứu tài liệu nội bộ (Internal Knowledge Assistant) cho một hệ thống hỗ trợ kỹ thuật/CSKH nhiều sản phẩm.
- **Đặc thù bài toán & Lý do chọn giải pháp:** Phần lớn câu hỏi người dùng thực tế là factoid đơn giản (tra cứu 1 sự thật trong 1 tài liệu) — Flat RAG/Hybrid RAG đã đủ tốt và rẻ hơn nhiều, đúng như quan sát thực nghiệm ở mục 4 (Flat RAG thắng ở nhóm factoid). GraphRAG chỉ đáng đầu tư cho phân nhóm câu hỏi cross-doc/multi-hop thực sự cần nối thông tin qua nhiều tài liệu (ví dụ: "sản phẩm A và B có xung đột cấu hình gì đã từng ghi nhận qua các bản vá trước đây?") — nên chọn kiến trúc **Hybrid**: Flat RAG làm mặc định, GraphRAG kích hoạt có điều kiện khi câu hỏi được phân loại multi-hop/cross-doc (tương tự router trong lab này).
- **Cấu trúc Node & Relation dự kiến:**
  - Nodes: `Product`, `Component`, `Issue/Bug`, `ConfigParameter`, `Person` (kỹ sư phụ trách)
  - Relations: `DEPENDS_ON`, `CAUSED_BY`, `FIXED_IN`, `CONFLICTS_WITH`, `OWNED_BY`
- **Chiến lược xử lý Super-node & Entity Resolution:** Các `Component` dùng chung nhiều sản phẩm (ví dụ thư viện lõi) sẽ là super-node tự nhiên — áp dụng cùng chính sách degree-cap + ưu tiên cạnh mới nhất như trong lab, nhưng bổ sung ràng buộc theo `product_version` thay vì chỉ theo thời gian (đúng bài học từ ca lỗi `G5000-26`). Entity Resolution ưu tiên Manual Aliases (danh sách tên sản phẩm/module chính thức) trước, vector ANN + lexical guard chỉ dùng cho các biến thể viết tắt không nằm trong danh sách chuẩn.

---

## 🎯 TỰ ĐÁNH GIÁ
| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | 4 | Nắm rõ kiến trúc Hybrid Retrieval và trade-off thực nghiệm; cần tìm hiểu thêm về Community Detection (bonus chưa triển khai). |
| Khả năng kiểm soát AI Coding Agent | 4 | Đã đọc và xác minh từng root cause trước khi chấp nhận patch của Agent (không patch mù), đặc biệt với các lỗi ẩn trong cơ chế retry/model-switch. |
| Chất lượng đồ thị tri thức xây dựng | 3 | Đồ thị còn thưa (`EXTRACTION_MAX_CHUNKS=400`) khiến GraphRAG thua Flat RAG về chất lượng trong benchmark — điểm cần cải thiện rõ nhất. |
| Khả năng phân tích và debug hệ thống | 5 | Truy vết thành công chuỗi lỗi liên hoàn (scope bug, string-match bug, JSON-mode leak) đến tận root cause thay vì chỉ retry mù. |
