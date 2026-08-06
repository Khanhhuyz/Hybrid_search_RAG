"""Fit confidence probabilities from reviewed RAG outcomes.

Input JSON shape:
{"cases": [{"evidence_score": 0.8, "groundedness_score": 1.0, "is_correct": true}]}
"""
import argparse
import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def expected_calibration_error(labels, probabilities, bins=10):
    total, error = len(labels), 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [
            i for i, value in enumerate(probabilities)
            if low <= value < high or (index == bins - 1 and value == 1.0)
        ]
        if not members:
            continue
        accuracy = sum(labels[i] for i in members) / len(members)
        confidence = sum(probabilities[i] for i in members) / len(members)
        error += len(members) / total * abs(accuracy - confidence)
    return error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    cases = payload.get("cases", payload)
    usable = [case for case in cases if "is_correct" in case]
    if len(usable) < 20:
        raise SystemExit("At least 20 reviewed cases are required for calibration")
    features = [
        [float(case["evidence_score"]), float(case["groundedness_score"])]
        for case in usable
    ]
    labels = [int(bool(case["is_correct"])) for case in usable]
    if len(set(labels)) != 2:
        raise SystemExit("Calibration data must contain both correct and incorrect answers")

    model = LogisticRegression(class_weight="balanced", random_state=42)
    model.fit(features, labels)
    probabilities = model.predict_proba(features)[:, 1].tolist()
    result = {
        "version": 1,
        "features": ["evidence_score", "groundedness_score"],
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "training_cases": len(usable),
        "brier_score": round(float(brier_score_loss(labels, probabilities)), 6),
        "expected_calibration_error": round(
            expected_calibration_error(labels, probabilities), 6
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
