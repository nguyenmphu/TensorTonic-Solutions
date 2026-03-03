import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    return np.trapezoid(np.array(tpr), np.array(fpr))
