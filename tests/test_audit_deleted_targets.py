import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_deleted_targets.py"
SPEC = importlib.util.spec_from_file_location("audit_deleted_targets", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_audit_deleted_targets_filters_and_classifies_failures():
    records = [
        {
            "image": "a",
            "target_component_eliminated": True,
            "target_quality_gt": 0.5,
            "target_quality": 0.05,
            "clutter_prob": 0.9,
        },
        {
            "image": "b",
            "target_component_eliminated": True,
            "target_quality_gt": 0.1,
            "target_quality": 0.05,
            "clutter_prob": 0.9,
        },
        {"image": "kept", "target_component_eliminated": False},
    ]

    audit = MODULE.audit_records(records)

    assert audit["num_deleted_targets"] == 2
    assert audit["failure_category_counts"] == {
        "A_quality_head_underestimates_target": 1,
        "B_quality_target_semantically_low": 1,
    }
    assert [item["image"] for item in audit["deleted_targets"]] == ["a", "b"]


def test_audit_deleted_targets_rejects_incompatible_candidate_records():
    try:
        MODULE.audit_records([{"image": "old-schema"}])
    except ValueError as error:
        assert "target_component_eliminated" in str(error)
    else:
        raise AssertionError("incompatible records must not silently report zero deletion")
