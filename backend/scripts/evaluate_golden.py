"""Run deterministic RAG/graph regression metrics against a golden JSON set."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.evaluator import Evaluator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    cases = payload.get("cases", payload)
    report = Evaluator.evaluate_batch(cases)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    try:
        print(rendered)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((rendered + "\n").encode("utf-8"))
    thresholds = payload.get("thresholds", {})
    if args.thresholds:
        thresholds.update(json.loads(args.thresholds.read_text(encoding="utf-8")))
    failures = [
        f"{metric}={report['aggregate'].get(metric, 0):.4f} < {minimum:.4f}"
        for metric, minimum in thresholds.items()
        if report["aggregate"].get(metric, 0.0) < float(minimum)
    ]
    if failures:
        print("Quality gate failed: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
