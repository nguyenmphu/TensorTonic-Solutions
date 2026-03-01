import numpy as np

def roc_metric(y_true, y_score, threshold):
    n = y_true.shape[0]
    tp = np.sum(y_true * (y_score >= threshold))
    tn = np.sum((1 - y_true) * (y_score < threshold))
    fn = np.sum(y_true) - tp
    fp = n - tp - tn - fn
    print(threshold, tp, tn, fn, tp)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return (tpr, fpr)
    

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """
    # Write code here
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_score, list):
        y_score = np.array(y_score)
    thresholds = [np.inf]
    thresholds.extend(sorted(set(y_score), reverse=True))
    tprs, fprs = [], []
    for threshold in thresholds:
        tpr, fpr = roc_metric(y_true, y_score, threshold)
        tprs.append(tpr)
        fprs.append(fpr)
    return np.array(fprs), np.array(tprs), np.array(thresholds)
