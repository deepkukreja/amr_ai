from scripts.evaluation.broad_benchmark import cohort_metrics, select_coverage


def test_partial_labels_exclude_unverified_classes_and_incomplete_groups():
    assert select_coverage({"dog", "car", "cat"}, {"truck", "bird"}, {"car", "bird"}) == [
        "cat", "dog", "truck",
    ]


def test_unsupported_and_unverified_predictions_do_not_inflate_false_positives():
    frame = {"coverage": ["dog", "cat", "truck"], "boxes": [{"label": "dog", "box": [0, 0, 1, 1]}]}
    preds = [{"label": label, "box": [0, 0, 1, 1], "confidence": .9}
             for label in ("dog", "cat", "car", "truck")]
    report = cohort_metrics([(frame, preds)], {"dog", "cat", "car"}, .25)
    assert report["dog"]["tp"] == 1
    assert report["cat"]["fp"] == 1
    assert report["car"]["scored_images"] == 0
    assert "truck" not in report


def test_duplicate_prediction_is_false_positive_and_ap_uses_confidence_order():
    truth = {"coverage": ["dog"], "boxes": [{"label": "dog", "box": [0, 0, .5, .5]}]}
    preds = [{"label": "dog", "box": box, "confidence": conf}
             for box, conf in [([.6, .6, 1, 1], .95), ([0, 0, .5, .5], .9), ([0, 0, .5, .5], .8)]]
    result = cohort_metrics([(truth, preds)], {"dog"}, .25)["dog"]
    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 0)
    assert result["ap50"] == .5


def test_open_images_test_split_preserves_attribution_and_coverage(tmp_path, monkeypatch):
    from scripts.data import freeze_broad_benchmark as sampler

    image_id = "0123456789abcdef"
    monkeypatch.setattr(sampler, "ROOT", tmp_path)
    monkeypatch.setattr(sampler, "OPEN_LABELS", {"Dog": "dog"})
    (tmp_path / "class-descriptions-boxable.csv").write_text("d,Dog\n")
    (tmp_path / "test-images-with-rotation.csv").write_text(
        "ImageID,Subset,Rotation,License,Author,OriginalLandingURL\n"
        f"{image_id},test,0,https://creativecommons.org/licenses/by/2.0/,Author,https://example.org/photo\n"
    )
    (tmp_path / "test-annotations-human-imagelabels-boxable.csv").write_text(
        f"ImageID,LabelName,Confidence\n{image_id},d,1\n"
    )
    (tmp_path / "test-annotations-bbox.csv").write_text(
        "ImageID,LabelName,XMin,YMin,XMax,YMax,IsGroupOf,IsDepiction,IsInside\n"
        f"{image_id},d,0,0,1,1,0,0,0\n"
    )

    def download(key, folder, split):
        assert (key, split) == (image_id, "test")
        path = folder / (key + ".jpg")
        path.write_bytes(b"image")
        return path

    monkeypatch.setattr(sampler, "download_image", download)
    [frame] = sampler.open_images(tmp_path / "images", tmp_path, "test", 1)
    assert frame["cohort"] == "open_images_v5_test"
    assert frame["coverage"] == ["dog"]
    assert (frame["author"], frame["source"]) == ("Author", "https://example.org/photo")
