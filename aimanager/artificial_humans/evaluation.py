import pandas as pd
import torch as th
import os
from aimanager.utils.array_to_df import add_labels
from aimanager.utils.utils import make_dir
from sklearn.metrics import (
    mean_absolute_error,
    accuracy_score,
    log_loss,
)


def eval_model(model, data):
    metrics = []
    strategies = ["greedy", "sampling"]
    for strategy in strategies:
        if strategy == "greedy":
            y_pred, y_pred_proba = model.predict_pure(data, sample=False)
        elif strategy == "sampling":
            y_pred, y_pred_proba = model.predict_pure(data, sample=True)

        # mask y_true, y_pred, y_pred_proba
        mask = data["mask"]
        y_true = th.masked_select(data["y"], mask)
        y_true = y_true.detach().cpu().numpy()

        y_pred = th.masked_select(y_pred, mask)
        y_pred = y_pred.detach().cpu().numpy()

        n_levels = y_pred_proba.shape[-1]
        y_pred_proba = th.masked_select(y_pred_proba, mask.unsqueeze(-1))
        y_pred_proba = y_pred_proba.reshape(-1, n_levels)
        y_pred_proba = y_pred_proba.detach().cpu().numpy()

        metrics += [
            {
                "name": "mean_absolute_error",
                "value": mean_absolute_error(y_true, y_pred),
                "strategy": strategy,
            },
            {
                "name": "accuracy",
                "value": accuracy_score(y_true, y_pred),
                "strategy": strategy,
            },
        ]
    # log loss is independent of the sampling strategy
    metrics += [
        {
            "name": "log_loss",
            "value": log_loss(y_true, y_pred_proba, labels=list(range(n_levels))),
        },
    ]
    return metrics


class Recorder:
    def __init__(self):
        self.metrics = []

    def set_labels(self, **labels):
        self.labels = labels

    def rec(self, value, name="loss"):
        self.metrics.append(dict(name=name, value=value, **self.labels))

    def rec_many(self, metrics):
        metrics = [{**m, **self.labels} for m in metrics]
        self.metrics += metrics

    def save(self, output_path, labels, job_id="all"):
        self._save_metric(self.metrics, output_path, "metrics", labels, job_id)

    @staticmethod
    def _save_metric(rec, output_path, metric_name, labels, job_id="all"):
        metric_path = os.path.join(output_path, metric_name)
        make_dir(metric_path)
        df = pd.DataFrame(rec)
        df = add_labels(df, {**labels, "job_id": job_id})
        df.to_parquet(os.path.join(metric_path, f"{job_id}.parquet"))
