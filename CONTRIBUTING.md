# Contributing to Mini-GraphRAG (GRAG)

Thank you for your interest in contributing to **Mini-GraphRAG**! We welcome contributions from developers, researchers, and open-source enthusiasts.

---

## 🛠️ How to Contribute

1. **Fork the Repository:** Create your own fork of [`quinc-fptu/mini-graphrag`](https://github.com/quinc-fptu/mini-graphrag).
2. **Clone your Fork:**
   ```bash
   git clone https://github.com/your-username/mini-graphrag.git
   cd mini-graphrag
   ```
3. **Create a Feature Branch:**
   ```bash
   git checkout -b feature/amazing-feature
   ```
4. **Make Your Changes & Test (Mandatory):**
   - **Backend Unit Tests:** `cd backend && python -m unittest discover -s tests`
   - **Frontend Build Check:** `cd frontend && npm run build`
5. **Commit & Push:**
   - Follow Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`).
   ```bash
   git commit -m "feat: add amazing feature"
   git push origin feature/amazing-feature
   ```
6. **Open a Pull Request:** Submit a PR against the `main` branch. GitHub Actions CI will automatically run tests against your PR.

---

## 💬 Code Style & Mandatory Standards

1. **Unit Testing:** All new backend features or bug fixes MUST include corresponding unit tests inside `backend/tests/`.
2. **Pinned Dependencies:** Never add unpinned packages to `backend/requirements.txt`. Always pin minimum/maximum versions (e.g. `package>=1.0.0,<2.0.0`).
3. **CI Pipeline Compliance:** PRs with failing GitHub Actions CI checks (`.github/workflows/ci.yml`) will not be merged.
4. **Code Style:**
   - Use **Python 3.11+** with type hints and Pydantic v2 schemas.
   - Follow **2-space indentation** for TypeScript/React/CSS and **4-space** for Python.
5. **License Notice:** This project uses the **CC BY-NC 4.0** (Non-Commercial) license. Ensure contributions do not include proprietary or commercial-only code.

