"""Choose a detection operating threshold using validation counts only."""
import math


def select_threshold(rows, *, minimum_recall, selection_split):
    """Maximize F1 subject to a recall floor; expose an infeasible floor."""
    if selection_split != 'validation':
        raise ValueError('Threshold selection requires validation data')
    if not math.isfinite(minimum_recall) or not 0 <= minimum_recall <= 1:
        raise ValueError('Minimum recall must be finite and between zero and one')
    scored = []
    for row in rows:
        threshold = row['threshold']
        tp, fp, fn = (row[key] for key in ('tp', 'fp', 'fn'))
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Confidence threshold must be finite and normalized')
        if any(not isinstance(value, int) or value < 0 for value in (tp, fp, fn)):
            raise ValueError('Detection counts must be nonnegative integers')
        scored.append({**row, 'precision': tp/(tp+fp) if tp+fp else 0.0,
                       'recall': tp/(tp+fn) if tp+fn else 0.0,
                       'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0})
    totals = {r['tp']+r['fn'] for r in scored}
    if len(totals) != 1 or not next(iter(totals)):
        raise ValueError('A consistent, positive ground truth count is required')
    feasible = [r for r in scored if r['recall'] >= minimum_recall]
    if feasible:
        chosen = max(feasible, key=lambda r: (r['f1'], r['recall'], -r['threshold']))
    else:
        chosen = max(scored, key=lambda r: (r['recall'], r['f1'], -r['threshold']))
    return {**chosen, 'minimum_recall': minimum_recall,
            'recall_floor_met': bool(feasible), 'selection_split': selection_split}
