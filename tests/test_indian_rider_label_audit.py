from pathlib import Path

from scripts.data.indian_rider_label_audit import choose


def test_selection_uses_distinct_clips_including_previous_picks():
    frames = [
        {"image": Path(name), "clip_id": group, "anchor": "rider", "light": light}
        for name, group, light in (
            ("a.jpg", "already-used", "night"),
            ("b.jpg", "b", "daytime"), ("c.jpg", "b", "night"),
            ("d.jpg", "d", "dusk"), ("e.jpg", "e", "night"),
        )
    ]
    selected = choose(frames, {"rider": 3}, {"already-used"})
    assert {row["clip_id"] for row in selected} == {"b", "d", "e"}
    assert choose(list(reversed(frames)), {"rider": 3}, {"already-used"}) == selected


def test_paired_geometry_comparison_uses_the_same_predictions_for_both_labels():
    from scripts.evaluation.score_indian_rider_label_audit import paired_matches

    legacy = [[.1, .1, .2, .2]]
    revised = [[.1, .1, .3, .3]]
    assert paired_matches(legacy, revised, revised) == {
        "legacy_matches": 0, "revised_matches": 1, "both": 0,
        "legacy_only": 0, "revised_only": 1, "neither": 0,
    }
