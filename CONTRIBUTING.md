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
4. **Make Your Changes & Test:**
   - Run backend syntax checks: `python -m uvicorn app.main:app`
   - Run frontend build checks: `cd frontend && npm run build`
5. **Commit & Push:**
   ```bash
   git commit -m "feat: add amazing feature"
   git push origin feature/amazing-feature
   ```
6. **Open a Pull Request:** Submit a PR against the `main` branch with a clear description of your changes.

---

## 💬 Code Style & Conventions

- Use **Python 3.11+** with type hints and Pydantic v2 schemas.
- Follow **2-space indentation** for TypeScript/React/CSS and **4-space** for Python.
- Format commits using Conventional Commits (`feat:`, `fix:`, `docs:`, `style:`).
