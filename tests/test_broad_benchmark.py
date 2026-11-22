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
