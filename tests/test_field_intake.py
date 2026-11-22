import csv
import json

import pytest
from PIL import Image

from scripts.data.field_intake import freeze_batch, stage_batch
from scripts.evaluation.evaluate_broad_benchmark import load_frames


def captures(tmp_path, splits=("development", "holdout"), same_session=False, same_image=False):
    path = tmp_path / "captures.csv"
    fields = ["image", "site_id", "session_id", "camera", "capture_utc", "scene", "lighting", "split"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, split in enumerate(splits):
            rgba = (100, 50, 30) if same_image else (100, index * 50, 30)
            image = f"frame-{index}.png"
            Image.new("RGB", (12, 9), color=rgba).save(tmp_path / image)
            writer.writerow(dict(image=image, site_id="yard", session_id="run0" if same_session else f"run{index}",
                                 camera="front", capture_utc=f"2026-11-22T10:00:0{index}Z",
                                 scene="loading_yard", lighting="day", split=split))
    return path


def review_all(batch, example):
    for path in sorted((batch / "annotations").glob("*.json")):
        record = json.loads(path.read_text())
        record.update(reviewed=True, coverage=["pedestrian", "car"], boxes=example)
        path.write_text(json.dumps(record))


def test_same_capture_session_cannot_cross_splits(tmp_path):
    with pytest.raises(ValueError, match="session"):
        stage_batch(captures(tmp_path, same_session=True), tmp_path / "batch")
    assert not (tmp_path / "batch").exists()


def test_exact_frame_duplicate_is_rejected_even_if_sessions_differ(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        stage_batch(captures(tmp_path, same_image=True), tmp_path / "batch")


def test_pending_annotation_cannot_be_frozen_as_negative(tmp_path):
    batch = tmp_path / "batch"
    stage_batch(captures(tmp_path), batch)
    with pytest.raises(ValueError, match="review"):
        freeze_batch(batch, "holdout")
    assert not (batch / "holdout.json").exists()


def test_reviewed_empty_image_keeps_explicit_coverage_and_freezes_holdout_only(tmp_path):
    batch = tmp_path / "batch"
    stage_batch(captures(tmp_path), batch)
    review_all(batch, [])
    result = freeze_batch(batch, "holdout")
    frozen = json.loads(result.read_text())
    assert frozen["scope"] == "field holdout"
    assert len(frozen["frames"]) == 1
    assert frozen["frames"][0]["session_id"] == "run1"
    assert frozen["frames"][0]["coverage"] == ["car", "pedestrian"]
    assert frozen["frames"][0]["boxes"] == []


def test_label_requires_declared_review_coverage(tmp_path):
    batch = tmp_path / "batch"
    stage_batch(captures(tmp_path), batch)
    review_all(batch, [{"label": "rider", "box": [.1, .1, .8, .9]}])
    with pytest.raises(ValueError, match="coverage"):
        freeze_batch(batch, "holdout")


def test_changed_image_bytes_prevent_freeze(tmp_path):
    batch = tmp_path / "batch"
    stage_batch(captures(tmp_path), batch)
    review_all(batch, [])
    row = json.loads((batch / "capture_manifest.json").read_text())["frames"][1]
    (batch / row["image"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        freeze_batch(batch, "holdout")


def test_evaluator_accepts_sealed_field_batch_using_frozen_label_masks(tmp_path):
    batch = tmp_path / "batch"
    stage_batch(captures(tmp_path), batch)
    frame = sorted((batch / "annotations").glob("*.json"))[1]
    review = json.loads(frame.read_text())
    review.update(reviewed=True, coverage=["pedestrian", "rider", "car"],
                  boxes=[{"label": "rider", "box": [.1, .1, .8, .9]}])
    frame.write_text(json.dumps(review))
    result = freeze_batch(batch, "holdout")
    with pytest.raises(ValueError, match="holdout"):
        load_frames(result)
    loaded = next(iter(load_frames(result, allow_field_holdout=True).values()))
    assert loaded["cohort"] == "field_loading_yard"
    assert loaded["coverage"] == ["car", "human", "pedestrian", "rider"]
    assert [obj["label"] for obj in loaded["boxes"]] == ["rider", "human"]
