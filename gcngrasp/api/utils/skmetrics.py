from functools import partial

import numpy as np
import sklearn.metrics as skm

METRICS_HARD = {
    "acc": skm.accuracy_score,
    "f1": skm.f1_score,
    "pre": skm.precision_score,
    "rec": skm.recall_score,
}
METRICS_HARD_MACRO = {
    "acc": skm.accuracy_score,
    "f1": partial(skm.f1_score, average="macro"),
    "pre": partial(skm.precision_score, average="macro"),
    "rec": partial(skm.recall_score, average="macro"),
}
METRICS_SOFT = {
    "AP": skm.average_precision_score,
    "AUC": skm.roc_auc_score,
}


def precision_greedy(idx: np.ndarray,
                     y_true: np.ndarray,
                     y_pred: np.ndarray) -> float:
    data = {
        k: (v[:, 0].astype(int), v[:, 1])
        for k, v in group_items(idx, np.stack([y_true, y_pred], axis=1)).items()
    }
    data = {k: ls for k, ls in data.items() if np.any(ls[0])}

    labels = np.array([l[s.argmax()] for l, s in data.values()])
    return np.mean(labels == 1)


def bin_cls_optimal_thresh(y_true: np.ndarray,
                           y_score: np.ndarray,
                           monitor: str = "acc",
                           mode: str = "max",
                           candidates: np.ndarray = np.arange(0.1, 0.91, 0.05)
                           ) -> np.ndarray:
    """ Calculate optimal threshold for binary classification. """
    y_true, y_score = y_true.flatten(), y_score.flatten()
    y_pred = y_score >= candidates[:, None].repeat(len(y_score), axis=1)
    monitor = METRICS_HARD[monitor]
    metrics = np.array([monitor(y_true, pred) for pred in y_pred])
    i = np.argmax(metrics) if mode == "max" else np.argmin(metrics)
    return candidates[metrics == metrics[i]]


def cls_metrics(y_true: np.ndarray,
                y_score: np.ndarray,
                y_pred: np.ndarray,
                mode: str) -> dict:
    """ Calculate classification metrics. """
    mode = ["binary", "multiclass", "multilabel"].index(mode)
    ret = {}

    for name, func in (METRICS_HARD if mode == 0 else METRICS_HARD_MACRO).items():
        ret[name] = func(y_true, y_pred)

    if mode == 1:
        y_true = np.eye(y_score.shape[-1], dtype=int)[y_true]

    for name, func in METRICS_SOFT.items():
        ret[name] = func(y_true, y_score)
    return ret


def group_items(idx: np.ndarray,
                items: np.ndarray) -> dict:
    idx = idx.tolist() if idx.ndim == 1 else map(tuple, idx.tolist())
    ret = {}
    for i, x in zip(idx, items):
        ret.setdefault(i, []).append(x)
    return {k: np.array(v) for k, v in ret.items()}


def mAP_score(idx: np.ndarray,
              y_true: np.ndarray,
              y_pred: np.ndarray,
              ret_org: bool = False,
              skip_undef: bool = True) -> float:
    data = {
        k: (v[:, 0].astype(int), v[:, 1])
        for k, v in group_items(idx, np.stack([y_true, y_pred], axis=1)).items()
    }
    if skip_undef: data = {k: ls for k, ls in data.items() if np.any(ls[0])}

    aps = {k: skm.average_precision_score(*ls) for k, ls in data.items()}
    return aps if ret_org else np.mean(list(aps.values()))


if __name__ == '__main__':
    y_true = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.8, 0.7, 0.2, 0.3, 0.6, 0.5, 0.4, 0.4])
    print(bin_cls_optimal_thresh(y_true, y_score))
