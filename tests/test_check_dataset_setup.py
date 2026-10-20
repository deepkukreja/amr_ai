"""Unit tests for the dataset setup checker."""

from scripts import check_dataset_setup as checker


def test_split_values_accepts_a_single_path():
    assert checker.split_values("images/train") == ["images/train"]


def test_split_values_filters_empty_and_non_string_entries():
    result = checker.split_values(
        ["images/train", "", "images/val", None, 42]
    )
    assert result == ["images/train", "images/val"]


def test_resolve_target_uses_base_for_relative_paths(tmp_path):
    base = tmp_path / "dataset"
    result = checker.resolve_target("images/train", base)

    assert result == (base / "images/train").resolve()


def test_resolve_target_handles_absolute_paths(tmp_path):
    target = (tmp_path / "absolute_images").resolve()

    assert checker.resolve_target(str(target), tmp_path) == target


def test_read_manifest_resolves_paths_relative_to_manifest(tmp_path):
    manifest_dir = tmp_path / "dataset"
    manifest_dir.mkdir()
    manifest = manifest_dir / "train.txt"
    manifest.write_text(
        "./images/frame.jpg\n\n# ignored comment\n",
        encoding="utf-8",
    )

    references, error = checker.read_manifest(manifest)

    assert error is None
    assert references == [
        (manifest_dir / "images/frame.jpg").resolve()
    ]


def test_inspect_split_detects_missing_manifest_images(tmp_path):
    dataset = tmp_path / "dataset"
    image_dir = dataset / "images"
    image_dir.mkdir(parents=True)

    (image_dir / "present.jpg").touch()
    (dataset / "train.txt").write_text(
        "images/present.jpg\nimages/missing.jpg\n",
        encoding="utf-8",
    )

    findings = checker.inspect_split(
        "train.txt", dataset, tmp_path
    )

    assert len(findings) == 1
    assert findings[0]["state"] == "MISSING IMAGES"
    assert findings[0]["references"] == 2
    assert findings[0]["missing_images"] == 1


def test_inspect_split_reports_missing_target(tmp_path):
    dataset = tmp_path / "dataset"

    findings = checker.inspect_split(
        "images/val", dataset, tmp_path
    )

    assert len(findings) == 1
    assert findings[0]["state"] == "MISSING TARGET"


def test_discover_configs_skips_calibration_files(tmp_path):
    datasets = tmp_path / "datasets"
    example = datasets / "example"
    calibration = datasets / "calibration_files"
    example.mkdir(parents=True)
    calibration.mkdir()

    (example / "data.yaml").write_text(
        "train: images/train\nval: images/val\n",
        encoding="utf-8",
    )
    (calibration / "camera.yaml").write_text(
        "camera_matrix: [1, 0, 0]\n",
        encoding="utf-8",
    )

    configs, errors, skipped = checker.discover_dataset_configs(
        tmp_path
    )

    assert len(configs) == 1
    assert configs[0]["file"] == example / "data.yaml"
    assert errors == []
    assert skipped == 1
