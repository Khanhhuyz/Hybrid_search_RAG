# 🛡️ Agent & Development Stability Guide

Tài liệu này tổng hợp nguyên nhân và các quy tắc để ngăn ngừa sự cố **"Cancellation / Server Restart"** (Tự động hủy tác vụ / Khởi động lại dịch vụ nền) trong quá trình phát triển ứng dụng với AI Assistant.

---

## 1. Nguyên nhân gây ra "Cancellation / Server Restart"

1. **Xung đột File Lock giữa Hot-Reload và AI Tool Edits:**
   * Khi các lệnh như `uvicorn --reload` (FastAPI) hoặc `next dev` (Turbopack) đang chạy ngầm trong IDE, trình theo dõi tệp (`WatchFiles` / `Chokidar`) sẽ liên tục lắng nghe thay đổi.
   * Khi AI thực hiện thao tác sửa tệp (`replace_file_content` hoặc `write_to_file`), hệ thống OS sẽ khóa tệp tạm thời. Nếu AI vừa sửa vừa chạy lệnh cùng lúc, watcher sẽ bị treo và kích hoạt lệnh khởi động lại tiến trình của IDE Extension.

2. **Tiến trình ngầm (Background Task) chạy quá lâu:**
   * Các lệnh như `run_command` nếu giữ tiến trình dev server chạy liên tục trong nền của IDE runner có thể bị cạn kiệt tài nguyên bộ nhớ đệm (buffer overflow) khi có hàng ngàn dòng log console tuôn ra.

3. **Xung đột Qdrant Storage / SQLite Lock (Local Mode):**
   * Nếu script test bên ngoài cố truy cập trực tiếp vào `data/qdrant_storage` hoặc `data/grag.db` trong khi server FastAPI đang giữ kết nối exclusive lock, Python sẽ bị ném lỗi `PermissionError / AlreadyLocked`.

---

## 2. Quy tắc & Giải pháp Khắc phục (Best Practices)

### 📌 Quy tắc dành cho AI Agent
- **Không giữ Dev Server ngầm khi thực hiện sửa mã nguồn đa tệp (Multi-file edits):**
  Tắt hoặc ngưng kích hoạt `run_command` chạy background dev server trong lúc đang sửa nhiều file liên tiếp. Chỉ bật lại dev server sau khi đã hoàn tất sửa đổi.
- **Tận dụng Qdrant Cloud hoặc Server-mode:**
  Kết nối Qdrant qua Cloud URL (`QDRANT_URL`) thay vì dùng file cục bộ `path` khi phát triển song song nhiều script.
- **Luôn kiểm tra trạng thái Task trước khi kết thúc:**
  Dùng `manage_task` để theo dõi hoặc dọn dẹp các task bị treo.

### 📌 Quy tắc dành cho Developer (Người dùng)
- **Chạy Dev Server ở Terminal ngoài:**
  Khuyên dùng Terminal riêng của máy tính (PowerShell/CMD bên ngoài) để chạy `python -m uvicorn app.main:app` và `npm run dev`. Điều này giúp IDE độc lập 100% với môi trường thực thi và không bao giờ bị gián đoạn session.
- **Xóa cache khi gặp sự cố:**
  Nếu gặp lỗi Turbopack hoặc Next.js dev server bị kẹt, chỉ cần xóa thư mục `.next` hoặc `.venv/__pycache__`.

---

## 3. Bản tóm tắt nhanh (Quick Rules Checklist)
- [x] Đã cấu hình Qdrant Cloud để tránh `Portalocker` file lock.
- [x] Đã tạo `backend/Dockerfile` & `frontend/Dockerfile` cho môi trường chuẩn hóa.
- [x] Khuyên dùng Terminal ngoài cho Dev Servers nếu muốn giữ IDE phản hồi tức thì.
