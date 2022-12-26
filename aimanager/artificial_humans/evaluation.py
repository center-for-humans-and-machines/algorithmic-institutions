import pandas as pd
import torch as th
import os
from aimanager.artificial_humans.metrics import (
    create_metrics,
    calc_log_loss,
)
from aimanager.utils.array_to_df import add_labels
from aimanager.utils.utils import make_dir


def eval_set(self, model, data, **add_labels):
    metrics = []
    strategies = ["greedy", "sampling"]
    for strategy in strategies:
        if strategy == "greedy":
            y_pred, y_pred_proba = model.predict_pure(data, sample=False)
        elif strategy == "sampling":
            y_pred, y_pred_proba = model.predict_pure(data, sample=True)

        mask = data["mask"]
        y_true = th.masked_select(data["y"], mask)
        y_true = y_true.detach().cpu().numpy()

        y_pred = th.masked_select(y_pred, mask)
        y_pred = y_pred.detach().cpu().numpy()

        n_levels = y_pred_proba.shape[-1]
        y_pred_proba_ = th.masked_select(y_pred_proba, mask.unsqueeze(-1))
        y_pred_proba_ = y_pred_proba_.reshape(-1, n_levels)

        y_pred_proba_ = y_pred_proba_.detach().cpu().numpy()

        metrics.append(
            calc_log_loss(
                y_true,
                y_pred_proba_,
                n_levels=n_levels,
                **add_labels,
                **self.labels,
                strategy=strategy,
            )
        )

        metrics += create_metrics(
            y_true, y_pred, strategy=strategy, **add_labels, **self.labels
        )
    return metrics


class Recorder:
    def __init__(self):
        self.metrics = []

    def set_labels(self, **labels):
        self.labels = labels

    def rec(self, value, name="loss"):
        self.metrics.append(dict(name=name, value=value, **self.labels))

    def rec_many(self, metrics):
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
