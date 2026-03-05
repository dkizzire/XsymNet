import os
from typing import Optional, Sequence, Tuple, Union, List, Dict, Any
import numpy as np
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from xgboost import XGBClassifier, Booster, XGBModel
from xgboost import dask as dxgb
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask.array as da
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

ArrayLike = Union[np.ndarray, list]

class XGBoostWrapper:
    def __init__(self, **xgb_kwargs):
        """
        Wrapper for XGBClassifier for XRD ML.
        Accepts standard XGBClassifier kwargs.
        """
        self.xgb_kwargs = xgb_kwargs
        self.label_encoder = LabelEncoder()
        self._model = None
        self.fitted = False

    def _build_xgb_classifier(self):
        return XGBClassifier(**self.xgb_kwargs)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: Optional[ArrayLike] = None,
        eval_set: Optional[Sequence[Tuple[ArrayLike, ArrayLike]]] = None,
        early_stopping_rounds: Optional[int] = None,
        eval_metric: Optional[str] = None,
        verbose_eval: Optional[Union[bool, int]] = True,
        **fit_kwargs
    ):
        """
        Fit the XGBoost model.
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Build classifier
        clf = self._build_xgb_classifier()

        # Set eval_metric if provided
        if eval_metric is not None:
            clf.set_params(eval_metric=eval_metric)

        # Handle early stopping
        if early_stopping_rounds is not None and eval_set is not None:
            clf.callbacks = [
                EarlyStopping(
                    rounds=early_stopping_rounds,
                    save_best=True,
                    maximize=False
                )
            ]

        # Map verbose_eval → verbose
        verbose = verbose_eval if verbose_eval is not None else True

        # Prepare fit parameters (only allowed kwargs)
        fit_params = {
            "sample_weight": sample_weight,
            "eval_set": eval_set,
            "verbose": verbose
        }
        fit_params.update(fit_kwargs)

        # Fit model
        clf.fit(X, y_encoded, **fit_params)

        # Store trained model
        self._model = clf
        self.fitted = True
        return self

    def predict(self, X: ArrayLike):
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        y_pred_encoded = self._model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X: ArrayLike):
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self._model.predict_proba(X)

    def get_model(self):
        return self._model


def top_k_accuracy(y_true, y_proba, k: int = 3) -> float:
    """
    Compute top-k accuracy given predicted probabilities.

    Args:
        y_true (array-like): Ground truth labels (int-encoded).
        y_proba (ndarray): Predicted probabilities, shape (n_samples, n_classes).
        k (int): Number of top predictions to consider.

    Returns:
        float: Top-k accuracy.
    """
    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
    match = [y in top_k for y, top_k in zip(y_true, top_k_preds)]
    return np.mean(match)

def evaluate_classification(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
    expt: Optional[np.ndarray] = None,
    labels: Optional[Sequence[int]] = None,
    class_names: Optional[Sequence[str]] = None,
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Compute and display classification metrics including top-k accuracy.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        y_proba (ndarray, optional): Predicted probabilities, shape (n_samples, n_classes).
        labels (list, optional): Label order for confusion matrix.
        class_names (list, optional): Display names for classes.
        plot (bool): Whether to show a confusion matrix heatmap.

    Returns:
        dict: Metrics including accuracy, top-k, F1, precision, recall,
              confusion matrix, and classification report.
    """
    results = {
        "accuracy_top1": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=class_names, zero_division=0
        ),
    }

    # Top-k metrics (only if probabilities are available)
    if y_proba is not None:
        results["accuracy_top3"] = top_k_accuracy(y_true, y_proba, k=3)
        results["accuracy_top5"] = top_k_accuracy(y_true, y_proba, k=5)

    # Optionally plot confusion matrix
    if plot:
        cm = results["confusion_matrix"]
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names if class_names is not None else labels,
            yticklabels=class_names if class_names is not None else labels,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    return results



