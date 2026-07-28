# 🚀 Mini-GraphRAG Optimization Roadmap

Tài liệu này tổng hợp toàn bộ các phương án có thể tối ưu cho hệ thống Mini-GraphRAG hiện tại. Các tối ưu này được phân loại từ trải nghiệm người dùng, độ chính xác của RAG, tối ưu tài nguyên, cho đến đầu vào/đầu ra (I/O).

---

## 1. Tối ưu Đầu vào / Đầu ra (I/O) & Xử lý Dữ liệu
*Đóng vai trò quan trọng khi người dùng tải lên nhiều file hoặc file dung lượng lớn.*

*   **Streaming File Upload (Input):** Hiện tại FastAPI đang nhận toàn bộ file vào RAM/Disk trước khi xử lý. Có thể tối ưu bằng cách xử lý Streaming Upload (đọc từng chunk của file và lưu thẳng xuống đĩa) để tiết kiệm RAM khi xử lý file > 100MB.
*   **Asynchronous Document Parsing (Input):** Việc bóc tách PDF/Docx có thể làm block Event Loop của FastAPI nếu file quá lớn. Cần đưa tác vụ `DocumentProcessor` vào ThreadPool (sử dụng `run_in_executor`) để không làm nghẽn các request khác.
*   **Intelligent Text Chunking (Input):**
    *   Thay vì dùng `RecursiveCharacterTextSplitter` (cắt theo ký tự cứng), áp dụng **Semantic Chunking** (cắt theo sự thay đổi ngữ nghĩa) hoặc **Structural Chunking** (cắt theo Heading của Markdown/PDF) sẽ giúp LLM đọc hiểu context dễ hơn.
*   **Streaming LLM Response (Output):**
    *   Hiện tại, API `/chat` đợi LLM sinh xong mới trả về (`stream: False`). Việc áp dụng **Server-Sent Events (SSE)** để stream từng chữ (token) về Frontend sẽ giảm độ trễ cảm nhận từ hàng chục giây xuống chỉ còn dưới 1 giây.
*   **Caching & Response Deduplication (Output):**
    *   Sử dụng Redis hoặc bộ nhớ đệm (LRU) lưu lại các câu hỏi đã trả lời. Dùng so sánh vector (Vector Similarity) giữa các câu hỏi để trả về ngay kết quả từ Cache nếu trùng lặp ý tưởng > 95%.

---

## 2. Tối ưu Độ chính xác (Accuracy) & Kiến trúc RAG
*Để câu trả lời chính xác, thông minh và đầy đủ ngữ cảnh hơn.*

*   **Reciprocal Rank Fusion (RRF) cho Hybrid Search:**
    *   Hiện tại Semantic Context và Graph Context được lấy độc lập và nối chuỗi thẳng vào Prompt. RRF sẽ giúp đánh giá lại (re-rank) và hòa trộn hai tập kết quả này, đẩy những thông tin thực sự giá trị nhất lên đầu để tránh làm nhiễu LLM (Context Window Limitation).
*   **Graph-based Query Expansion:**
    *   Thay vì chỉ tìm Entity trong câu hỏi, hệ thống có thể đi dạo (traverse) qua các Entity lân cận (depth=2, 3) để tìm ra các "từ khóa ẩn". Dùng các từ khóa này để truy vấn ngược lại Vector Store, giúp trả lời các câu hỏi mang tính khái quát (Multi-hop reasoning).
*   **Self-Querying & Intent Router:**
    *   Sử dụng một LLM call nhẹ (hoặc classifier) đầu vào để phân loại: "Câu này có cần gọi Knowledge Graph không, hay chỉ cần Semantic, hay chỉ là chào hỏi phiếm?". Điều này tiết kiệm rất nhiều token và thời gian xử lý khi không cần thiết.
*   **Graph Filtering & Edge Weighting:**
    *   Thêm cơ chế đánh trọng số (Weight) và phân loại (Type) cho các liên kết (Edge) trên Graph. Khi lấy Context, ta có thể chỉ lấy các liên kết "mạnh" nhất thay vì lấy toàn bộ lân cận gây loãng thông tin.

---

## 3. Tối ưu Trải nghiệm Người dùng (Frontend / UX)
*Giúp ứng dụng mượt mà, chuyên nghiệp và phản hồi tức thời.*

*   **Tối ưu Render Knowledge Graph:**
    *   Force Graph hiện tại sẽ bị sụt giảm FPS nếu số lượng nodes/edges > 1000. Có thể áp dụng **Clustering** (gom nhóm các node gần nhau khi zoom out), **Pagination/Lazy Load**, hoặc nâng cấp hẳn lên **WebGL 3D Force Graph**.
*   **Citations Highlight (Trích dẫn tương tác):**
    *   Thay vì chỉ liệt kê trích dẫn, khi hover hoặc click vào Citation, ứng dụng tự động nhảy (scroll) tới đoạn text gốc trong file, hoặc làm nổi bật Entity tương ứng ngay trên Knowledge Graph (Cross-highlighting).
*   **Streaming UI & Markdown Auto-render:**
    *   Hiển thị Markdown dần dần theo Streaming Response. Có loading skeletons (UI chờ) mượt mà cho các phần tử Citations và Graph Data.

---

## 4. Tối ưu Hiệu suất Backend & Tài nguyên
*Làm cho hệ thống chịu tải tốt hơn và tận dụng tối đa phần cứng.*

*   **Batching & Message Queue cho Graph Extraction:**
    *   Bước trích xuất Graph (dùng LLM bóc tách Node/Edge) tốn rất nhiều thời gian. Không nên chạy trực tiếp trên API request. Cần tách riêng thành một **Background Worker** (như Celery, RabbitMQ hoặc RQ). Khi upload, trả về trạng thái "Đang xử lý ngầm", giúp API không bao giờ bị timeout.
*   **Thay thế SQLite bằng CSDL chuyên dụng (Long-term):**
    *   Việc dùng SQLite + `asyncio.Lock` để tránh Deadlock hiện tại chỉ phù hợp cho demo cá nhân. Để scale, có thể cân nhắc chuyển sang **PostgreSQL (với pgvector)** cho Vector Store và **Neo4j** cho Graph Database để khai thác tận cùng sức mạnh của Graph.
*   **Tối ưu hóa Ollama Concurrency:**
    *   Quản lý kết nối tới Ollama chặt chẽ hơn: thêm Semaphore cho các tác vụ IO lớn, xử lý Timeout Retries và Keep-Alive connection tốt hơn bằng `httpx.AsyncClient` pool toàn cục.

---

## 🎯 Tóm lược Thứ tự Khuyến nghị Thực hiện (Action Plan)
1. **Quick Wins (Dễ & Hiệu quả ngay):** Streaming Response (SSE) cho endpoint Chat + Tối ưu hóa UI để render streaming.
2. **Core RAG Upgrades (Cải thiện não bộ):** Triển khai RRF Reranking và Graph-based Query Expansion.
3. **Architecture Upgrades (Cải thiện hệ cơ bắp):** Đưa toàn bộ Logic Document Processing & Graph Extraction vào Background Task/Queue (ví dụ: Celery/Redis).
