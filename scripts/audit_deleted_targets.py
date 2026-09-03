#!/usr/bin/env python3
"""Audit target components removed by an SCA-CCRR candidate decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


AUDIT_FIELDS = (
    "image",
    "candidate_id",
    "action_component_id",
    "matched_gt_id",
    "clutter_prob",
    "target_quality",
    "target_quality_gt",
    "target_guard",
    "target_guard_gt",
    "guard_valid",
    "guard_allow",
    "quality_allow",
    "soft_guard_allow",
    "soft_quality_allow",
    "final_gate",
    "risk_score",
    "gate",
    "action_area",
    "proposal_area",
    "proposal_is_fallback",
    "proposal_to_component_iou",
    "raw_peak",
    "raw_mean",
    "scale_features",
    "geometry_features",
    "proposal_relation_feature_norm",
    "action_mask_feature_norm",
    "proposal_box_xyxy",
    "action_box_xyxy",
    "action_centroid_yx",
    "protected_by",
)


def _finite_float(record: Mapping[str, Any], key: str) -> float | None:
    value = record.get(key)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def classify_failure(
    record: Mapping[str, Any],
    *,
    low_quality_gt: float = 0.2,
    low_quality_prediction: float = 0.2,
    high_clutter_probability: float = 0.8,
) -> str:
    """Map a deleted target to the A/B/C taxonomy in the TG-SCA plan."""

    quality_gt = _finite_float(record, "target_quality_gt")
    quality = _finite_float(record, "target_quality")
    clutter = _finite_float(record, "clutter_prob")
    if quality_gt is not None and quality_gt < low_quality_gt:
        return "B_quality_target_semantically_low"
    if quality is not None and quality < low_quality_prediction:
        return "A_quality_head_underestimates_target"
    if (
        clutter is not None
        and clutter >= high_clutter_probability
        and quality is not None
        and quality >= low_quality_prediction
    ):
        return "C_conflicting_clutter_and_target_evidence"
    return "other_requires_replay"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            records.append(record)
    return records


def audit_records(
    records: Iterable[Mapping[str, Any]],
    *,
    low_quality_gt: float = 0.2,
    low_quality_prediction: float = 0.2,
    high_clutter_probability: float = 0.8,
) -> dict[str, Any]:
    records = list(records)
    for index, record in enumerate(records):
        if "target_component_eliminated" not in record:
            raise ValueError(
                "candidate record "
                f"{index} lacks required field 'target_component_eliminated'"
            )
    deleted = [
        record for record in records if bool(record["target_component_eliminated"])
    ]
    required_deleted_fields = (
        "image",
        "target_quality_gt",
        "target_quality",
        "clutter_prob",
    )
    for index, record in enumerate(deleted):
        missing = [key for key in required_deleted_fields if key not in record]
        if missing:
            raise ValueError(
                f"deleted-target record {index} lacks required fields: "
                + ", ".join(missing)
            )
    audited: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}
    for record in deleted:
        category = classify_failure(
            record,
            low_quality_gt=low_quality_gt,
            low_quality_prediction=low_quality_prediction,
            high_clutter_probability=high_clutter_probability,
        )
        category_counts[category] = category_counts.get(category, 0) + 1
        selected = {key: record[key] for key in AUDIT_FIELDS if key in record}
        selected["failure_category"] = category
        audited.append(selected)
    audited.sort(
        key=lambda item: (
            str(item.get("image", "")),
            int(item.get("candidate_id", -1)),
        )
    )
    return {
        "schema_version": "mshnet-tg-sca-deleted-target-audit/v1",
        "thresholds": {
            "low_quality_gt": float(low_quality_gt),
            "low_quality_prediction": float(low_quality_prediction),
            "high_clutter_probability": float(high_clutter_probability),
        },
        "num_deleted_targets": len(audited),
        "failure_category_counts": category_counts,
        "deleted_targets": audited,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--low-quality-gt", type=float, default=0.2)
    parser.add_argument("--low-quality-prediction", type=float, default=0.2)
    parser.add_argument("--high-clutter-probability", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"candidate JSONL does not exist: {args.input}")
    for name in (
        "low_quality_gt",
        "low_quality_prediction",
        "high_clutter_probability",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must lie in [0,1]")
    payload = audit_records(
        load_jsonl(args.input),
        low_quality_gt=args.low_quality_gt,
        low_quality_prediction=args.low_quality_prediction,
        high_clutter_probability=args.high_clutter_probability,
    )
    payload["source"] = {
        "path": str(args.input.resolve()),
        "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
