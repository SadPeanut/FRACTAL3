"""
evaluation.py
-------------
Model quality metrics computed from a confusion matrix.

MLlib only offers overall accuracy, so per-class IoU and mIoU are computed
here. The heavy part (counting every (true class, predicted class) pair) runs
distributed in Spark; only the small K x K matrix reaches the driver.
"""
import numpy as np


def confusion_matrix(pred_df, n_classes):
    """K x K matrix of counts: rows = true class, columns = predicted class."""
    rows = pred_df.groupBy("label", "prediction").count().collect()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for r in rows:
        cm[int(r["label"]), int(r["prediction"])] = r["count"]  # not r.count, which is a method of Row
    return cm


def metrics_from_confusion(cm, class_names):
    """Overall accuracy, per-class IoU and mIoU from a confusion matrix.

    IoU(c) = TP / (TP + FP + FN). A class that never occurs, neither in the
    reference nor in the  predictions (possible on very small samples), has no
    defined IoU: it is reported as None and left out of the mIoU.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp   # predicted as c, but really another class
    fn = cm.sum(axis=1) - tp   # really c, but predicted as another class
    denominator = tp + fp + fn

    iou = [float(tp[i] / denominator[i]) if denominator[i] > 0 else None for i in range(len(tp))]
    defined = [v for v in iou if v is not None]
    return {
        "overall_accuracy": float(tp.sum() / cm.sum()),
        "mIoU": float(np.mean(defined)),
        "n_classes_in_mIoU": len(defined),
        "per_class_iou": dict(zip(class_names, iou)),
        "support": dict(zip(class_names, cm.sum(axis=1).tolist())),  # true points per class
        "confusion_matrix": cm.tolist(),
    }


def evaluate(model, df, class_names):
    """Predict on `df` (must contain `label` + the feature columns) and compute all metrics."""
    predictions = model.transform(df).select("label", "prediction")
    cm = confusion_matrix(predictions, len(class_names))  # this action triggers the prediction
    return metrics_from_confusion(cm, class_names)
