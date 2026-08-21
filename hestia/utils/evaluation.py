from collections import defaultdict
from typing import Dict

import numpy as np

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (matthews_corrcoef,
                             accuracy_score, f1_score, log_loss,
                             precision_score, recall_score, mean_squared_error,
                             mean_absolute_error, roc_auc_score, r2_score)


def _log_loss(preds, truths):
    return log_loss(truths, preds, normalize=True)


def _pcc(preds, truths):
    return pearsonr(preds, truths)[0]


def _spcc(preds, truths):
    return spearmanr(preds, truths)[0]


def _f1_weighted(preds, truths):
    return f1_score(preds, truths, average='weighted')


def _recall(preds, truths):
    return recall_score(preds, truths, zero_division=True)


def _tp(preds, truths):
    return ((preds == 1) & (truths == 1)).sum()


def _tn(preds, truths):
    return ((preds == 0) & (truths == 0)).sum()


def _fp(preds, truths):
    return ((preds == 1) & (truths == 0)).sum()


def _fn(preds, truths):
    return ((preds == 0) & (truths == 1)).sum()


CLASSIFICATION_METRICS = {
    'mcc': matthews_corrcoef,
    'acc': accuracy_score,
    'f1': f1_score,
    'f1_weighted': _f1_weighted,
    'precision': precision_score,
    'recall': _recall,
    'auroc': roc_auc_score,
    'log_loss': _log_loss,
    'tp': _tp,
    'tn': _tn,
    'fp': _fp,
    'fn': _fn

}

REGRESSION_METRICS = {
    'mse': mean_squared_error,
    'mae': mean_absolute_error,
    'pcc': _pcc,
    'spcc': _spcc,
    'r2': r2_score
}


def evaluate(preds, truth, pred_task) -> Dict[str, float]:
    result = {}
    if pred_task == 'reg':
        metrics = REGRESSION_METRICS
    else:
        metrics = CLASSIFICATION_METRICS

    for key, value in metrics.items():
        if key in ['auroc'] or pred_task == 'reg':
            t_pred = preds
        else:
            t_pred = preds > 0.5
        try:
            result[key] = value(truth, t_pred)
        except ValueError:
            result[key] = 0.0
        if np.isnan(result[key]):
            result[key] = 0.0
    return result
