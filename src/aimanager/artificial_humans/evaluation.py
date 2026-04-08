import pandas as pd
import torch as th
import os
from aimanager.utils.array_to_df import add_labels, using_multiindex
from aimanager.utils.utils import make_dir
from sklearn.metrics import (
    mean_absolute_error,
    accuracy_score,
    log_loss,
)


def create_confusion_matrix(model, data, y_name, labels):
    y_pred, y_pred_proba = model.predict_encoded(data, sample=False)
    if isinstance(y_pred_proba, dict):
        y_pred_proba = y_pred_proba[y_name]
    y_pred_proba = y_pred_proba.detach().cpu().numpy()

    mask = data["mask"]
    y_true = data["y"].detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    proba_df = using_multiindex(
        y_pred_proba, ["idx", "round_number", f"pred_{y_name}"]
    ).rename(columns={"value": "proba"})
    mask_df = using_multiindex(mask, ["idx", "round_number"]).rename(
        columns={"value": "valid"}
    )

    y_df = using_multiindex(y_true, ["idx", "round_number"]).rename(
        columns={"value": f"true_{y_name}"}
    )

    df = proba_df.merge(mask_df).merge(y_df)
    # df = df[df['valid']]
    df = add_labels(df, labels)
    return df


def _eval_single_head(model, data, mask, y_true_tensor, head=None):
    """Evaluate a single prediction head."""
    metrics = []
    strategies = ["greedy", "sampling"]
    for strategy in strategies:
        sample = strategy == "sampling"
        preds = model.predict_encoded(data, sample=sample)
        y_pred, y_pred_proba = preds

        if isinstance(y_pred, dict):
            name = head or model.y_name
            y_pred = y_pred[name]
            y_pred_proba = y_pred_proba[name]

        y_true = th.masked_select(y_true_tensor, mask)
        y_true = y_true.detach().cpu().numpy()

        y_pred = th.masked_select(y_pred, mask)
        y_pred = y_pred.detach().cpu().numpy()

        n_levels = y_pred_proba.shape[-1]
        y_pred_proba = th.masked_select(
            y_pred_proba, mask.unsqueeze(-1)
        )
        y_pred_proba = y_pred_proba.reshape(-1, n_levels)
        y_pred_proba = y_pred_proba.detach().cpu().numpy()

        head_label = {"head": head} if head else {}
        metrics += [
            {
                "name": "mean_absolute_error",
                "value": mean_absolute_error(y_true, y_pred),
                "strategy": strategy,
                **head_label,
            },
            {
                "name": "accuracy",
                "value": accuracy_score(y_true, y_pred),
                "strategy": strategy,
                **head_label,
            },
        ]
    metrics += [
        {
            "name": "log_loss",
            "value": log_loss(
                y_true, y_pred_proba,
                labels=list(range(n_levels)),
            ),
            **head_label,
        },
    ]
    return metrics


def eval_model(model, data, validity_target=None, validity_mask=None):
    metrics = _eval_single_head(
        model, data, data["mask"], data["y"],
        head="contribution" if validity_target is not None else None,
    )
    if validity_target is not None:
        metrics += _eval_single_head(
            model, data, validity_mask, validity_target,
            head="contribution_valid",
        )
    return metrics


class Recorder:
    def __init__(self):
        self.metrics = []

    def set_labels(self, **labels):
        self.labels = labels

    def rec(self, value, name="loss", **labels):
        self.metrics.append(dict(name=name, value=value, **self.labels, **labels))

    def rec_many(self, metrics, **labels):
        metrics = [{**m, **self.labels, **labels} for m in metrics]
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
