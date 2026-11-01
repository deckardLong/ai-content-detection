import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score

def compute_metrics(y_true, y_pred, y_prob_ai, types=None):
    """Label 1 = AI, 0 = Human. types (generated_type) -> also accuracy per text type."""
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_ai": p[1], "recall_ai": r[1], "f1_ai": f1[1],
        "precision_human": p[0], "recall_human": r[0], "f1_human": f1[0],
        "roc_auc": roc_auc_score(y_true, y_prob_ai) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    if types is not None:
        for t in np.unique(types):
            sel = types == t
            m[f"acc_{t}"] = accuracy_score(y_true[sel], y_pred[sel])
    return {k: float(v) for k, v in m.items()}
