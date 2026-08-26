import json

import numpy as np
import pytest

from scripts.fit_temperature import (
    CandidateSamples,
    apply_temperature,
    calibration_metrics,
    deterministic_image_split,
    extract_explicit_samples,
    fit_temperature,
    main,
)


def test_extract_uses_only_explicit_target_and_clutter_records():
    records = [
        {"name": "a", "score": 0.9, "label": "target", "label_id": 0},
        {"name": "a", "score": 0.2, "label": "clutter", "label_id": 1},
        {"name": "b", "score": 0.6, "label": "uncertain", "label_id": 2},
        {"name": "b", "score": 0.7},
    ]

    samples = extract_explicit_samples(records)

    np.testing.assert_allclose(samples.scores, [0.9, 0.2])
    np.testing.assert_array_equal(samples.labels, [0, 1])
    np.testing.assert_array_equal(samples.image_names, ["a", "a"])

    with pytest.raises(ValueError, match="contradictory"):
        extract_explicit_samples(
            [{"name": "a", "score": 0.9, "label": "target", "label_id": 1}]
        )


def test_image_split_is_deterministic_and_has_no_image_leakage():
    samples = CandidateSamples(
        scores=np.linspace(0.1, 0.9, 12),
        labels=np.tile([0, 1], 6),
        image_names=np.repeat([f"image_{index}" for index in range(6)], 2),
    )

    first = deterministic_image_split(samples, 0.5, seed=17)
    second = deterministic_image_split(samples, 0.5, seed=17)
    fit_samples, evaluation_samples, fit_names, evaluation_names = first

    assert fit_names == second[2]
    assert evaluation_names == second[3]
    assert set(fit_names).isdisjoint(evaluation_names)
    assert set(fit_samples.image_names).isdisjoint(evaluation_samples.image_names)
    assert len(fit_samples) + len(evaluation_samples) == len(samples)


def test_temperature_fit_improves_nll_and_preserves_ranking():
    # Correctly ordered but deliberately under-confident predictions.
    scores = np.array([0.65, 0.60, 0.55, 0.45, 0.40, 0.35])
    labels = np.array([0, 0, 0, 1, 1, 1])

    temperature = fit_temperature(scores, labels)
    calibrated = apply_temperature(scores, temperature)
    before = calibration_metrics(scores, labels)
    after = calibration_metrics(calibrated, labels)

    assert 0 < temperature < 1
    assert after["NLL"] <= before["NLL"]
    np.testing.assert_array_equal(
        np.argsort(-scores, kind="stable"),
        np.argsort(-calibrated, kind="stable"),
    )


def test_temperature_softens_probability_endpoints_consistently():
    calibrated = apply_temperature(np.asarray([0.0, 0.5, 1.0]), temperature=10.0)
    assert 0.0 < calibrated[0] < 0.5
    assert calibrated[1] == pytest.approx(0.5)
    assert 0.5 < calibrated[2] < 1.0


def test_cli_accepts_independent_validation_json_and_writes_invariance_note(
    tmp_path, capsys
):
    evaluation_records = [
        {"name": "eval_a", "score": 0.8, "label": "target", "label_id": 0},
        {"name": "eval_a", "score": 0.4, "label": "clutter", "label_id": 1},
        {"name": "eval_b", "score": 0.7, "label": "target", "label_id": 0},
        {"name": "eval_b", "score": 0.3, "label": "clutter", "label_id": 1},
        {"name": "eval_b", "score": 0.5, "label": "uncertain", "label_id": 2},
    ]
    validation_records = [
        {"name": "val_a", "score": 0.7, "label": "target", "label_id": 0},
        {"name": "val_a", "score": 0.3, "label": "clutter", "label_id": 1},
        {"name": "val_b", "score": 0.6, "label": "target", "label_id": 0},
        {"name": "val_b", "score": 0.4, "label": "clutter", "label_id": 1},
    ]
    evaluation_path = tmp_path / "test_candidates.json"
    validation_path = tmp_path / "val_candidates.json"
    output_path = tmp_path / "temperature.json"
    evaluation_path.write_text(
        json.dumps({"metadata": {}, "candidates": evaluation_records}),
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps({"metadata": {}, "candidates": validation_records}),
        encoding="utf-8",
    )

    main(
        [
            "--candidate-json",
            str(evaluation_path),
            "--val-json",
            str(validation_path),
            "--output-json",
            str(output_path),
            "--n-bins",
            "5",
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["split"]["mode"] == "independent_validation_file"
    assert report["calibration_fit"]["num_candidates"] == 4
    assert report["evaluation"]["num_candidates"] == 4
    assert report["detection_metric_invariance"]["ranking_preserved"] is True
    assert (
        report["detection_metric_invariance"][
            "can_improve_ranking_or_threshold_sweep_metrics"
        ]
        is False
    )
    assert {"ECE", "Brier", "NLL"} == set(report["evaluation"]["before"])
    assert {"ECE", "Brier", "NLL"} == set(report["evaluation"]["after"])
    assert "cannot improve ranking/FROC" in capsys.readouterr().err
