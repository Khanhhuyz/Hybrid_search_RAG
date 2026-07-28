# Project Workspace Rules for AI Agents

## ⛔ CRITICAL RULE: Prevent IDE Cancellation & Extension Crashes

- **STRICTLY PROHIBITED**: Never execute or leave long-running dev servers (`uvicorn --reload`, `npm run dev`) running in background tool calls (`run_command`) while performing multi-file code edits.
- **WHY**: Hot-reloading file watchers (`WatchFiles`/`Turbopack`) lock files during write operations, causing severe extension host timeouts and triggering `User cancelled agent execution` / `Server restart` events.
- **RECOMMENDED EXECUTION FLOW**:
  1. Make all code edits cleanly without background dev servers active.
  2. Run single-pass build/test commands (`npm run build` or short python tests) to verify correctness.
  3. Commit & push changes to Git.
  4. Only instruct the user to run dev servers in an external terminal or launch short-lived test commands when requested.

## 📋 General Stability & Execution Guidelines

- **Reference Stability Documentation:**
  Before debugging environment cancellations, inspect `.agents/doc/dev_stability_guide.md`.
- **Cross-reference Snapshot Data:**
  Cross-reference `.agents/doc/context-snapshot.md` with live code before modifying critical services.
- **Do Not Re-launch Background Servers:**
  Never auto-launch background dev servers unless explicitly requested by the user for a live preview test.

