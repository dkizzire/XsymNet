"""
xgboost_wrapper.py

Drop-in XGBoost wrapper tailored for XRD classification pipelines.

Requirements:
    pip install xgboost scikit-learn matplotlib joblib

If you use GPU, set use_gpu=True and ensure xgboost was installed with GPU support.

Example usage at the bottom of this file shows how to load .pt files saved from PyTorch
and call the wrapper.

Author: Dayton Kizzire
Date: 2025-09-23

"""

from typing import Optional, Union, Tuple, Dict, Any
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

import xgboost as xgb
from xgboost.callback import EarlyStopping

# Optional torch support to convert .pt files into numpy
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


class XGBoostWrapper:
    """
    Scikit-learn style wrapper for XGBoost with conveniences for XRD data.

    Methods:
        fit(X, y, X_val=None, y_val=None, sample_weight=None, eval_metric=None, **fit_kwargs)
        predict(X)
        predict_proba(X)
        save(path)
        load(path)
        plot_feature_importance(k=30, figsize=(10,6))
    """

    def __init__(
        self,
        num_class: Optional[int] = None,
        objective: Optional[str] = None,
        n_estimators: int = 300,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 0,
        n_jobs: int = -1,
        tree_method: Optional[str] = None,
        use_gpu: bool = False,
        verbosity: int = 1,
        **kwargs
    ):
        """
        Initialize the wrapper.

        Args:
            num_class: number of classes for multiclass. If None, will be inferred on fit.
            objective: xgboost objective. If None, set to 'multi:softprob' for multi-class, 'binary:logistic' for 2 classes.
            n_estimators, learning_rate, max_depth, subsample, colsample_bytree, reg_alpha, reg_lambda: xgboost params.
            tree_method: if None, defaults to 'hist' or 'gpu_hist' if use_gpu True.
            use_gpu: bool, prefer GPU training if available and xgboost supports it.
            verbosity: 0 (silent) -> 3 (debug)
            **kwargs: passed to xgb.XGBClassifier
        """
        # keep constructor minimal; real params set during fit after seeing y
        self.num_class = num_class
        self.objective = objective
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.tree_method = tree_method
        self.use_gpu = use_gpu
        self.verbosity = verbosity
        self.xgb_params_override = kwargs

        self._model = None
        self.label_encoder = LabelEncoder()
        self.fitted = False

    @staticmethod
    def _ensure_numpy(X: Union[np.ndarray, 'torch.Tensor', list]) -> np.ndarray:
        """Convert torch Tensor or list to numpy 2D array (n_samples, n_features)."""
        if _HAS_TORCH and isinstance(X, torch.Tensor):
            return X.detach().cpu().numpy()
        if isinstance(X, list):
            return np.asarray(X)
        return X

    def _prepare_objective(self, y: np.ndarray):
        """Set objective and num_class based on y if not already specified."""
        unique = np.unique(y)
        n_classes = unique.shape[0]
        self.num_class = self.num_class or n_classes
        if self.objective is None:
            if n_classes == 2:
                self.objective = "binary:logistic"
            else:
                self.objective = "multi:softprob"

    def _build_xgb_classifier(self):
        """Construct the xgboost classifier with current params."""
        tree_method = self.tree_method
        if self.use_gpu:
            # prefer gpu_hist if available
            try:
                # if tree_method not explicitly set, set to gpu_hist
                if tree_method is None:
                    tree_method = "gpu_hist"
            except Exception:
                pass
        else:
            if tree_method is None:
                tree_method = "hist"

        params = dict(
            objective=self.objective,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method=tree_method,
            verbosity=self.verbosity,
            use_label_encoder=False,  # disable deprecated label encoder warnings
        )
        params.update(self.xgb_params_override)
        # For multiclass, specify num_class if objective is multi:softprob
        if "multi:" in (self.objective or ""):
            params["num_class"] = self.num_class

        clf = xgb.XGBClassifier(**params)
        return clf

    # def fit(
    #     self,
    #     X: Union[np.ndarray, 'torch.Tensor', list],
    #     y: Union[np.ndarray, 'torch.Tensor', list],
    #     X_val: Optional[Union[np.ndarray, 'torch.Tensor', list]] = None,
    #     y_val: Optional[Union[np.ndarray, 'torch.Tensor', list]] = None,
    #     sample_weight: Optional[np.ndarray] = None,
    #     eval_metric: Optional[str] = None,
    #     early_stopping_rounds: Optional[int] = 10,
    #     test_size: float = 0.1,
    #     random_state: int = 0,
    #     verbose_eval: Optional[int] = 50,
    #     **fit_kwargs
    # ):
    #     """
    #     Fit the XGBoost model.

    #     Args:
    #         X, y: training data. Can be numpy or torch.Tensors or lists.
    #         X_val, y_val: optional explicit validation set. If provided, used for early stopping.
    #         sample_weight: optional sample weights for training set.
    #         eval_metric: evaluation metric, e.g. 'mlogloss' or 'merror'. If None uses sensible default.
    #         early_stopping_rounds: int or None.
    #         test_size: fraction of training set to split for validation if X_val is None.
    #         verbose_eval: frequency for printing evaluation metric (None to disable).
    #         **fit_kwargs: forwarded to model.fit()
    #     Returns:
    #         self
    #     """
    #     # Convert to numpy
    #     X = self._ensure_numpy(X)
    #     y = self._ensure_numpy(y)
    #     # Flatten labels if necessary
    #     y = y.ravel()

    #     # Encode labels
    #     y_encoded = self.label_encoder.fit_transform(y)
    #     self._prepare_objective(y_encoded)

    #     # Build classifier
    #     clf = self._build_xgb_classifier()

    #     # Prepare evaluation set
    #     eval_set = None
    #     evals_result = {}
    #     if X_val is not None and y_val is not None:
    #         X_val = self._ensure_numpy(X_val)
    #         y_val = self._ensure_numpy(y_val).ravel()
    #         y_val_enc = self.label_encoder.transform(y_val)
    #         eval_set = [(X_val, y_val_enc)]
    #     else:
    #         if test_size and test_size > 0:
    #             X_train, X_hold, y_train, y_hold = train_test_split(
    #                 X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    #             )
    #             X = X_train
    #             y_encoded = y_train
    #             eval_set = [(X_hold, y_hold)]
    #         # else use no validation

    #     # default eval_metric
    #     if eval_metric is None:
    #         if "multi:" in self.objective:
    #             eval_metric = "mlogloss"
    #         elif self.objective == "binary:logistic":
    #             eval_metric = "logloss"
    #         else:
    #             eval_metric = "logloss"

    #     # put eval_metric into classifier params instead of fit() for XGB ≥ 2.0
    #     clf.set_params(eval_metric=eval_metric)
        
    #     # Prepare callbacks
    #     callbacks = []
    #     if early_stopping_rounds is not None and eval_set is not None:
    #         callbacks.append(EarlyStopping(
    #             rounds=early_stopping_rounds,
    #             save_best=True,
    #             maximize=False
    #         ))

    #     # build fit params
    #     fit_params = dict(
    #         eval_set=eval_set,
    #         verbose=verbose_eval,
    #         callbacks=callbacks,
    #     )

    #     # merge user-provided fit kwargs
    #     fit_params.update(fit_kwargs)

    #     # Call fit
    #     if sample_weight is not None:
    #         clf.fit(X, y_encoded, sample_weight=sample_weight, **fit_params)
    #     else:
    #         clf.fit(X, y_encoded, **fit_params)

    #     self._model = clf
    #     self.fitted = True
    #     # store last evals
    #     self.evals_result_ = clf.evals_result() if hasattr(clf, "evals_result") else {}
    #     return self

    def fit(self, X, y, *, sample_weight=None, eval_set=None,
        early_stopping_rounds=None, eval_metric=None,
        verbose_eval=50, **fit_kwargs):

        # Encode labels if needed
        y_encoded = self.label_encoder.fit_transform(y)

        # Build classifier
        clf = self._build_xgb_classifier()

        # Assign eval_metric
        if eval_metric is not None:
            clf.set_params(eval_metric=eval_metric)

        # Handle early stopping using callbacks
        if early_stopping_rounds is not None:
            from xgboost.callback import EarlyStopping
            clf.callbacks = [EarlyStopping(rounds=early_stopping_rounds, save_best=True, maximize=False)]

        # Map verbose_eval → verbose
        verbose = verbose_eval if verbose_eval is not None else True

        # Build fit_params dict with only accepted arguments
        fit_params = {
            "sample_weight": sample_weight,
            "eval_set": eval_set,
            "verbose": verbose,
        }
        fit_params.update(fit_kwargs)

        # Fit the model
        clf.fit(X, y_encoded, **fit_params)

        self._model = clf
        self.fitted = True
        return self

    def predict(self, X: Union[np.ndarray, 'torch.Tensor', list]) -> np.ndarray:
        """Return decoded class labels (original labels)."""
        if not self.fitted:
            raise RuntimeError("Model is not fitted yet. Call fit(...) first.")
        X = self._ensure_numpy(X)
        y_pred_encoded = self._model.predict(X)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded.astype(int))
        return y_pred

    def predict_proba(self, X: Union[np.ndarray, 'torch.Tensor', list]) -> np.ndarray:
        """Return probability estimates for each class (rows sum to 1 for multiclass)."""
        if not self.fitted:
            raise RuntimeError("Model is not fitted yet. Call fit(...) first.")
        X = self._ensure_numpy(X)
        proba = self._model.predict_proba(X)
        return proba

    def score(self, X, y, sample_weight=None) -> float:
        """Return accuracy by default."""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

    def save(self, path: str):
        """
        Save the wrapper and model to a file using joblib.

        Args:
            path: path to file ending in .pkl or .joblib
        """
        # bundle relevant state
        state = {
            "model": self._model,
            "label_encoder": self.label_encoder,
            "params": {
                "num_class": self.num_class,
                "objective": self.objective,
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "reg_alpha": self.reg_alpha,
                "reg_lambda": self.reg_lambda,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "tree_method": self.tree_method,
                "use_gpu": self.use_gpu,
                "verbosity": self.verbosity,
                "xgb_params_override": self.xgb_params_override,
            },
            "fitted": self.fitted,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str) -> "XGBoostWrapper":
        """Load wrapper saved by save()."""
        state = joblib.load(path)
        obj = cls(**state["params"])
        obj._model = state["model"]
        obj.label_encoder = state["label_encoder"]
        obj.fitted = state.get("fitted", False)
        return obj

    def feature_importances(self) -> np.ndarray:
        """Return feature importances from the fitted model (gain-based importance if available)."""
        if not self.fitted:
            raise RuntimeError("Model is not fitted yet.")
        # XGBClassifier stores feature_importances_ or get_booster().get_score()
        if hasattr(self._model, "feature_importances_"):
            return np.array(self._model.feature_importances_)
        else:
            booster = self._model.get_booster()
            fs = booster.get_score(importance_type="gain")  # dict feature->importance
            # convert dict to array
            if not fs:
                return np.array([])
            max_idx = max(int(k[1:]) for k in fs.keys()) + 1
            arr = np.zeros(max_idx, dtype=float)
            for k, v in fs.items():
                idx = int(k[1:])
                arr[idx] = v
            return arr

    def plot_feature_importance(self, k: int = 30, figsize: Tuple[int, int] = (10, 6)):
        """Plot top-k feature importances (bars)."""
        fi = self.feature_importances()
        if fi.size == 0:
            raise RuntimeError("No feature importances available.")
        # get top-k
        idx = np.argsort(fi)[::-1][:k]
        labels = [f"f{i}" for i in idx]
        values = fi[idx]
        plt.figure(figsize=figsize)
        plt.bar(range(len(values)), values)
        plt.xticks(range(len(values)), labels, rotation=90)
        plt.ylabel("importance")
        plt.title(f"Top {len(values)} feature importances")
        plt.tight_layout()
        plt.show()

    #
    # Convenience helpers for XRD preprocessing
    #
    @staticmethod
    def log_transform_intensity(X: Union[np.ndarray, 'torch.Tensor'], offset: float = 1e-6) -> np.ndarray:
        """
        Apply a log10 transform to intensities to dampen very large peaks.
        Keeps shape unchanged. Adds offset to avoid log(0).
        """
        X = X.detach().cpu().numpy() if _HAS_TORCH and isinstance(X, torch.Tensor) else np.asarray(X)
        # keep original dtype float32/64
        return np.log10(X + offset)

    @staticmethod
    def standardize(X: Union[np.ndarray, 'torch.Tensor'], mean: Optional[np.ndarray] = None,
                    std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Standardize features per-column (feature-wise).
        Returns (X_standardized, mean, std).
        If mean/std provided they will be used (for applying same transform to test data).
        """
        X = X.detach().cpu().numpy() if _HAS_TORCH and isinstance(X, torch.Tensor) else np.asarray(X)
        if mean is None:
            mean = np.mean(X, axis=0)
        if std is None:
            std = np.std(X, axis=0)
            std[std == 0] = 1.0
        Xs = (X - mean) / std
        return Xs, mean, std


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Example: Loading a PyTorch .pt file where each row is a pattern (n_samples, n_features)
    # and labels are in a separate array.
    # Replace paths with your data.

    # NOTE: if you don't have torch installed, adjust to load numpy arrays instead.
    example_pt_path = "patterns.pt"     # expected shape (n_samples, n_features)
    example_labels_path = "labels.npy"  # shape (n_samples,)

    # Simple simulated demo if files don't exist:
    if not os.path.exists(example_pt_path) or not os.path.exists(example_labels_path):
        print("Demo: generating synthetic data for quick smoke test...")
        n_samples = 500
        n_features = 2048
        n_classes = 10
        X_demo = np.random.rand(n_samples, n_features).astype(np.float32) * 1e4  # large intensities
        y_demo = np.random.randint(0, n_classes, size=(n_samples,))
    else:
        if _HAS_TORCH and example_pt_path.endswith(".pt"):
            X_demo = torch.load(example_pt_path).detach().cpu().numpy()
        else:
            X_demo = np.load(example_pt_path)
        y_demo = np.load(example_labels_path)

    # small preprocessing example: log transform then standardize
    X_demo = XGBoostWrapper.log_transform_intensity(X_demo)
    X_demo, mean, std = XGBoostWrapper.standardize(X_demo)

    # create wrapper
    wrapper = XGBoostWrapper(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.6,
        use_gpu=False,  # set True if your xgboost supports GPU
        random_state=42,
        n_jobs=8,
    )

    # Fit with 10% holdout for early stopping
    wrapper.fit(X_demo, y_demo, early_stopping_rounds=20, verbose_eval=10)

    # Metrics
    acc = wrapper.score(X_demo, y_demo)
    print(f"Training accuracy (approx): {acc:.4f}")

    # Save model
    wrapper.save("xgb_wrapper_demo.joblib")
    print("Saved to xgb_wrapper_demo.joblib")