import numpy as np
from itertools import combinations
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

def mcnemar_pair(y, pred_a, pred_b):
    a_ok, b_ok = pred_a == y, pred_b == y
    table = [[int(np.sum(a_ok & b_ok)),  int(np.sum(a_ok & ~b_ok))],
             [int(np.sum(~a_ok & b_ok)), int(np.sum(~a_ok & ~b_ok))]]
    res = mcnemar(table, exact=True)   # kiểm định nhị thức chính xác
    return {'a_only_correct': table[0][1], 'b_only_correct': table[1][0],
            'p_value': float(res.pvalue)}

def compare_all(preds: dict, alpha=0.05):
    """preds: {tên_mô_hình: {'y':..., 'pred':...}} -> danh sách so sánh từng cặp."""
    rows = []
    for a, b in combinations(preds.keys(), 2):
        assert np.array_equal(preds[a]['y'], preds[b]['y']), 'Test set không khớp'
        r = mcnemar_pair(preds[a]['y'], preds[a]['pred'], preds[b]['pred'])
        rows.append({'model_a': a, 'model_b': b, **r})

    # 4 mô hình = 6 cặp -> cần hiệu chỉnh đa so sánh
    reject, p_adj, _, _ = multipletests([r['p_value'] for r in rows],
                                        alpha=alpha, method='holm')
    for r, pa, rej in zip(rows, p_adj, reject):
        r['p_holm'] = float(pa)
        r['significant'] = bool(rej)
    return rows