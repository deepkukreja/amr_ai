"""Score a frozen broad object development benchmark, respecting partial labels."""
from collections import defaultdict


def select_coverage(positive, negative, incomplete):
    return sorted((positive | negative) - incomplete)


def iou(a, b):
    intersection = (max(0, min(a[2], b[2]) - max(a[0], b[0])) *
                    max(0, min(a[3], b[3]) - max(a[1], b[1])))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def outcomes(truth, predictions, threshold):
    """Assign each detection to at most one truth box, in score order."""
    claimed = defaultdict(set)
    for score, frame, box in predictions:
        overlap, match = max(
            ((iou(box, target), idx) for idx, target in enumerate(truth[frame])
             if idx not in claimed[frame]), default=(0, -1),
        )
        matched = overlap >= threshold
        if matched:
            claimed[frame].add(match)
        yield score, int(matched)


def average_precision(matches, support):
    if not support:
        return None
    correct = 0
    points = []
    for count, (_, matched) in enumerate(matches, 1):
        if matched:
            correct += 1
            points.append(correct / count)
    best = 0
    area = 0
    for precision in reversed(points):
        best = max(best, precision)
        area += best
    return area / support


def cohort_metrics(frames, labels, confidence):
    results = {}
    for label in sorted(labels):
        truth = {}
        boxes = []
        for index, (frame, predictions) in enumerate(frames):
            if label not in frame["coverage"]:
                continue
            truth[index] = [obj["box"] for obj in frame["boxes"] if obj["label"] == label]
            boxes.extend((obj["confidence"], index, obj["box"]) for obj in predictions if obj["label"] == label)
        boxes.sort(key=lambda row: -row[0])
        support = sum(map(len, truth.values()))
        ranked = list(outcomes(truth, boxes, .5))
        selected = [matched for score, matched in ranked if score >= confidence]
        tp = sum(selected)
        fp = len(selected) - tp
        fn = support - tp
        ap = [average_precision(outcomes(truth, boxes, step / 100), support) for step in range(55, 100, 5)]
        ap50 = average_precision(ranked, support)
        results[label] = {
            "scored_images": len(truth), "support": support, "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / support if support else None,
            "ap50": ap50,
            "ap50_95": (ap50 + sum(ap)) / 10 if support else None,
        }
    return results
