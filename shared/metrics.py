"""Metrik evaluasi untuk audio anti-spoofing."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, matthews_corrcoef


def compute_eer(labels: np.ndarray, probs: np.ndarray) -> float:
    """
    Hitung Equal Error Rate (EER). Semakin kecil semakin baik.

    Args:
        labels: ground truth (0/1)
        probs:  probabilitas kelas positif (spoof)

    Returns:
        EER sebagai float [0, 1]
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def compute_metrics(labels: np.ndarray, preds: np.ndarray,
                    probs: np.ndarray) -> dict:
    """
    Hitung accuracy, ROC-AUC, EER, dan MCC sekaligus.

    Returns:
        dict dengan key: accuracy, roc_auc, eer, mcc
    """
    acc = float(np.mean(preds == labels))
    auc = float(roc_auc_score(labels, probs))
    eer = compute_eer(labels, probs)
    mcc = float(matthews_corrcoef(labels, preds))
    return {"accuracy": acc, "roc_auc": auc, "eer": eer, "mcc": mcc}
