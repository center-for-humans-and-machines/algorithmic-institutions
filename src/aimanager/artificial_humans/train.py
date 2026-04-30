import yaml
import sys
import os
import pandas as pd
import numpy as np
import random
import torch as th
import wandb
from aimanager.generic.data import create_torch_data, get_cross_validations
from aimanager.artificial_humans import AH_MODELS
from aimanager.artificial_humans.evaluation import (
    eval_model,
    Recorder,
    create_confusion_matrix,
)
from aimanager.utils.sanity import emit
from aimanager.utils.utils import make_dir
from itertools import permutations
from tqdm import tqdm


def shuffle_feature(data, feature_name):
    data = {**data}
    data[feature_name] = data[feature_name][th.randperm(len(data[feature_name]))]
    return data


def ablate_feature(data, feature_name):
    data = {**data}
    val = data[feature_name]
    mean = val.float().mean(0)
    if val.dtype == th.bool:
        mean = mean.round().bool()
    elif val.dtype in (th.int64, th.int32, th.long):
        mean = mean.round().to(val.dtype)
    data[feature_name] = mean.expand_as(val)
    return data


def batch_loader(data, batch_size):
    n = len(data["contribution"])
    all_idx = np.arange(n)
    all_idx = np.random.permutation(all_idx)
    n_batch = int(np.ceil(n / batch_size))
    for i in range(n_batch):
        batch_idx = all_idx[i * batch_size : (i + 1) * batch_size]
        if len(batch_idx) != batch_size:
            continue
        batch = {k: v[batch_idx] for k, v in data.items()}
        yield batch


def mask_data(data, mask, target, default_values):
    data = {**data}
    data[target + "_masked"] = data[target].clone()
    if mask.shape[0] != data[target + "_masked"].shape[0]:
        mask = mask.repeat(data[target + "_masked"].shape[0], 1)
    data[target + "_masked"][mask] = default_values[target]
    return data


def apply_mask_pattern(data, mask_pattern, y_name, mask_name, default_values):
    # the mask pattern defines which agents to predict (true: predict agent)
    # therefore we mask the target data for those agents that are predicted
    # we also compute the union between the new mask and the previous mask
    data = mask_data(data, mask_pattern, y_name, default_values)
    data[mask_name] = data[mask_name] & mask_pattern[:, :, np.newaxis]
    data["autoreg_mask"] = (
        th.ones_like(data[mask_name]) & mask_pattern[:, :, np.newaxis]
    )
    return data


def create_fully_connected(
    n_nodes, n_groups=1, n_agent_groups=1, device=th.device("cpu")
):
    return th.tensor(
        [
            [i + k * n_nodes, j + k * n_nodes]
            for k in range(n_groups)
            for i in range(n_nodes)
            for j in range(n_nodes)
            if i != j
        ],
        device=device,
    ).T


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main(config):
    seed = config["seed"]
    device = config["device"]
    autoregression = config["autoregression"]
    min_predicted = config.get("min_predicted", None)
    max_predicted = config.get("max_predicted", None)
    n_player = config["n_player"]
    n_agent_groups = config.get("n_groups", 1)
    n_cross_val = config["n_cross_val"]
    holdout_fold = config.get("holdout_fold", None)
    fraction_training = config["fraction_training"]
    model_name = config["model_name"]
    model_args = config["model_args"]
    optimizer_args = config["optimizer_args"]
    train_args = config["train_args"]
    shuffle_features = config.get("shuffle_features", [])
    ablate_features = config.get("ablate_features", [])
    mask_name = config["mask_name"]
    job_id = config["job_id"]
    data_file = config["data_file"]
    basedir = config["basedir"]
    output_dir = config["output_dir"]
    experiment_names = config["experiment_names"]
    labels = config.get("labels", {})

    if autoregression:
        if min_predicted is None:
            min_predicted = 1
        if max_predicted is None:
            max_predicted = n_player
        emit(
            "autoreg.bounds",
            {"min_predicted": min_predicted, "max_predicted": max_predicted},
        )

    switch_every = config.get("switch_every", None)

    emit(
        "config.snapshot",
        {
            "job_id": job_id,
            "data_file": data_file,
            "experiment_names": experiment_names,
            "mask_name": mask_name,
            "n_cross_val": n_cross_val,
            "n_player": n_player,
            "n_groups": n_agent_groups,
            "switch_every": switch_every,
            "model_name": model_name,
            "autoregression": autoregression,
            "y_name": model_args["y_name"],
            "y_levels": model_args["y_levels"],
            "hidden_size": model_args["hidden_size"],
            "add_rnn": model_args.get("add_rnn"),
            "add_edge_model": model_args.get("add_edge_model"),
            "add_global_model": model_args.get("add_global_model"),
            "x_features": [e["name"] for e in model_args["x_encoding"]],
            "optimizer_args": optimizer_args,
            "train_args": train_args,
            "shuffle_features": shuffle_features,
        },
    )

    model_dir = os.path.join(output_dir, "model")
    conf_dir = os.path.join(output_dir, "confusion_matrix")
    make_dir(model_dir)
    make_dir(conf_dir)

    model_path = os.path.join(model_dir, f"{job_id}.pt")
    conf_path = os.path.join(conf_dir, f"{job_id}.parquet")

    th.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    df_raw = pd.read_csv(os.path.join(basedir, data_file))
    raw_rows = len(df_raw)
    raw_episodes = (
        int(df_raw["episode_id"].nunique()) if "episode_id" in df_raw.columns else None
    )

    df = df_raw[df_raw["experiment_name"].isin(experiment_names)]
    after_filter_rows = len(df)
    after_filter_episodes = (
        int(df["episode_id"].nunique()) if "episode_id" in df.columns else None
    )

    emit(
        "dataset.csv_summary",
        {
            "raw_rows": raw_rows,
            "raw_episodes": raw_episodes,
            "after_filter_rows": after_filter_rows,
            "after_filter_episodes": after_filter_episodes,
        },
    )

    data, default_values = create_torch_data(df, switch_every=switch_every)

    emit(
        "dataset.tensor_shapes",
        {name: [list(t.shape), str(t.dtype)] for name, t in data.items()},
    )
    emit(
        "dataset.default_values",
        {k: (v.item() if hasattr(v, "item") else v) for k, v in default_values.items()},
    )

    n_rounds = data["contribution"].shape[2]
    if switch_every is not None:
        switch_rounds = [r for r in range(n_rounds) if (r + 1) % switch_every == 0]
    else:
        switch_rounds = list(range(n_rounds))
    emit("dataset.switch_rounds", switch_rounds)

    y_name = model_args["y_name"]
    mask_t = data[mask_name]
    y_t = data[y_name]
    if y_t.dtype == th.bool:
        pos = int((y_t & mask_t).sum().item())
        total = int(mask_t.sum().item())
        emit(
            "dataset.target_balance",
            {
                "positives": pos,
                "total": total,
                "fraction": round(pos / total, 6) if total else 0.0,
            },
        )
    else:
        counts = {}
        for c in y_t.unique().tolist():
            counts[int(c)] = int(((y_t == c) & mask_t).sum().item())
        emit(
            "dataset.target_balance",
            {"class_counts": counts, "total": int(mask_t.sum().item())},
        )

    emit(
        "dataset.first_episode_sample",
        {
            "contribution_ep0_p0_r0to5": data["contribution"][0, 0, :6].tolist(),
            "prev_contribution_ep0_p0_r0to5": data["prev_contribution"][
                0, 0, :6
            ].tolist(),
            "agent_group_ep0_p0_r0to5": data["agent_group"][0, 0, :6].tolist(),
            "prev_agent_group_ep0_p0_r0to5": data["prev_agent_group"][
                0, 0, :6
            ].tolist(),
        },
    )

    rec = Recorder()

    wandb_enabled = bool(os.environ.get("WANDB_API_KEY"))
    if wandb_enabled:
        wandb.init(
            config=config,
            name=job_id,
            group=os.path.basename(output_dir),
            tags=[f"{k}={v}" for k, v in labels.items()],
        )

    th_device = th.device(device)

    if autoregression:
        # the training mask defines which agents
        # are predicted (true: agent will be predicted)
        # correspondingly we mask inputs for these agents
        # here we define the base patter (i.e. the number of agents predicted)
        pattern = [
            [True] * i + [False] * (n_player - i)
            for i in range(min_predicted, max_predicted + 1)
        ]
        test_pattern = [
            [True] * i + [False] * (n_player - i) for i in range(1, n_player + 1)
        ]
    else:
        pattern = [[True] * n_player]
        test_pattern = [[True] * n_player]

    # we create all possible permutations of the training mask
    training_mask_pattern = list(set([pp for p in pattern for pp in permutations(p)]))
    training_mask_pattern = th.tensor(training_mask_pattern, dtype=th.bool)

    test_mask_pattern = list(set([pp for p in test_pattern for pp in permutations(p)]))
    test_mask_pattern = th.tensor(test_mask_pattern, dtype=th.bool)

    conf_m_all = []
    final_test_losses = []

    for i, train_data, test_data in get_cross_validations(
        data, n_cross_val, fraction_training, holdout_fold=holdout_fold
    ):
        model = AH_MODELS[model_name](
            default_values=default_values, autoregressive=autoregression, **model_args
        ).to(th_device)
        batch_size = train_args["batch_size"]
        batch_edge_index = create_fully_connected(
            n_player, n_groups=batch_size, n_agent_groups=n_agent_groups
        )
        train_edge_index = create_fully_connected(
            n_player,
            n_groups=train_data["contribution"].shape[0],
            n_agent_groups=n_agent_groups,
        )
        if test_data is not None:
            test_edge_index = create_fully_connected(
                n_player,
                n_groups=test_data["contribution"].shape[0],
                n_agent_groups=n_agent_groups,
            )
        y_name = model_args["y_name"]

        n_train_episodes = int(train_data["contribution"].shape[0])
        n_test_episodes = (
            int(test_data["contribution"].shape[0]) if test_data is not None else 0
        )
        emit(
            "cv.split_sizes",
            {"n_train_episodes": n_train_episodes, "n_test_episodes": n_test_episodes},
            fold=i,
        )

        if i == 0 or i is None:
            emit(
                "model.spec",
                {"n_params": int(sum(p.numel() for p in model.parameters()))},
            )
            emit(
                "cv.batch_edge_index_shape",
                {
                    "shape": list(batch_edge_index.shape),
                    "expected_edges": n_player * (n_player - 1) * batch_size,
                },
                fold=i,
            )
            emit(
                "train.loop_shape",
                {
                    "epochs": train_args["epochs"],
                    "batches_per_epoch": n_train_episodes // batch_size,
                },
                fold=i,
            )

        optimizer = th.optim.Adam(model.parameters(), **optimizer_args)
        loss_fn = th.nn.CrossEntropyLoss(reduction="none")
        sum_loss = 0
        n_steps = 0

        early_stopping_patience = train_args.get("early_stopping_patience")
        best_test_loss = float("inf")
        best_model_state = None
        epochs_without_improvement = 0
        last_test_log_loss = None

        pbar = tqdm(range(train_args["epochs"]))
        for e in pbar:
            rec.set_labels(cv_split=i, epoch=e)
            model.train()
            for j, b_data in enumerate(batch_loader(train_data, batch_size)):
                optimizer.zero_grad()
                p_idx = th.randint(0, len(training_mask_pattern), (batch_size,))
                b_data = apply_mask_pattern(
                    b_data,
                    training_mask_pattern[p_idx],
                    y_name,
                    mask_name,
                    default_values,
                )
                batch_data = model.encode(
                    b_data,
                    mask=mask_name,
                    edge_index=batch_edge_index,
                    device=th_device,
                )

                y_logit = model(batch_data).flatten(end_dim=-2)
                y_pred = y_logit.softmax(-1)
                y_true = batch_data["y_enc"].flatten(end_dim=-2)
                mask = batch_data["mask"].flatten()

                if (i == 0 or i is None) and e == 0 and j == 0:
                    emit(
                        "cv.first_batch_shapes",
                        {
                            "y_logit_shape": list(y_logit.shape),
                            "y_true_shape": list(y_true.shape),
                            "mask_sum": int(mask.sum().item()),
                            "mask_total": int(mask.numel()),
                        },
                        fold=i,
                    )

                loss = (
                    loss_fn(y_logit, y_true)
                    + (y_pred * y_pred.log()).sum(-1) * train_args["l1_entropy"]
                )

                loss = (loss * mask).sum() / mask.sum()

                loss.backward(retain_graph=True)

                if train_args.get("clamp_grad"):
                    for param in model.parameters():
                        param.grad.data.clamp_(
                            -train_args["clamp_grad"], train_args["clamp_grad"]
                        )

                optimizer.step()
                sum_loss += loss.item()
                n_steps += 1

            last_epoch = e == (train_args["epochs"] - 1)
            if (e % train_args["eval_period"] == 0) or last_epoch:
                avg_loss = sum_loss / n_steps
                rec.rec(value=avg_loss, set="train")

                # evalute on training data for all possible mask patterns
                for j, mask in enumerate(test_mask_pattern):
                    n_pred = mask.sum().item()
                    _d = apply_mask_pattern(
                        train_data, mask[np.newaxis], y_name, mask_name, default_values
                    )
                    _d = model.encode(
                        _d,
                        mask=mask_name,
                        edge_index=train_edge_index,
                        device=th_device,
                    )
                    metrics = eval_model(model, _d)
                    rec.rec_many(metrics, set="train", n_pred=n_pred, mask=j)

                test_log_loss = None
                perturb_log_loss = {}
                if test_data is not None:
                    # evalute on test data for all possible mask patterns
                    for j, mask in enumerate(test_mask_pattern):
                        n_pred = mask.sum().item()
                        _d = apply_mask_pattern(
                            test_data,
                            mask[np.newaxis],
                            y_name,
                            mask_name,
                            default_values,
                        )
                        _d = model.encode(
                            _d,
                            mask=mask_name,
                            edge_index=test_edge_index,
                            device=th_device,
                        )
                        metrics = eval_model(model, _d)
                        rec.rec_many(metrics, set="test", n_pred=n_pred, mask=j)
                        if j == 0:
                            for m in metrics:
                                if m["name"] == "log_loss":
                                    test_log_loss = m["value"]
                        for feats, fn, lbl in [
                            (shuffle_features, shuffle_feature, "shuffle_feature"),
                            (ablate_features, ablate_feature, "ablate_feature"),
                        ]:
                            for feat in feats:
                                _d = apply_mask_pattern(
                                    test_data,
                                    mask[np.newaxis],
                                    y_name,
                                    mask_name,
                                    default_values,
                                )
                                _d = fn(_d, feat)
                                _d = model.encode(
                                    _d,
                                    mask=mask_name,
                                    edge_index=test_edge_index,
                                    device=th_device,
                                )
                                metrics = eval_model(model, _d)
                                rec.rec_many(
                                    metrics,
                                    set="test",
                                    **{lbl: feat},
                                    n_pred=n_pred,
                                    mask=j,
                                )
                                if j == 0:
                                    for m in metrics:
                                        if m["name"] == "log_loss":
                                            perturb_log_loss[f"{lbl}_{feat}"] = m[
                                                "value"
                                            ]

                            # Leave-one-in: keep one feature intact,
                            # perturb all others
                            if len(feats) > 1:
                                loi_lbl = f"leave_one_in_{lbl}"
                                for feat in feats:
                                    others = [f for f in feats if f != feat]
                                    _d = apply_mask_pattern(
                                        test_data,
                                        mask[np.newaxis],
                                        y_name,
                                        mask_name,
                                        default_values,
                                    )
                                    for o in others:
                                        _d = fn(_d, o)
                                    _d = model.encode(
                                        _d,
                                        mask=mask_name,
                                        edge_index=test_edge_index,
                                        device=th_device,
                                    )
                                    metrics = eval_model(model, _d)
                                    rec.rec_many(
                                        metrics,
                                        set="test",
                                        **{loi_lbl: feat},
                                        n_pred=n_pred,
                                        mask=j,
                                    )

                if test_log_loss is not None:
                    last_test_log_loss = test_log_loss

                if wandb_enabled:
                    wandb_log = {
                        "epoch": e,
                        "fold": i,
                        f"fold_{i}/train/loss": avg_loss,
                    }
                    if test_log_loss is not None:
                        wandb_log[f"fold_{i}/test/log_loss"] = test_log_loss
                        for key, val in perturb_log_loss.items():
                            wandb_log[f"fold_{i}/test/log_loss__{key}"] = val
                    wandb.log(wandb_log)

                postfix = {"loss": f"{avg_loss:.4f}"}
                if test_log_loss is not None:
                    postfix["test_loss"] = f"{test_log_loss:.4f}"
                    if early_stopping_patience is not None:
                        if test_log_loss < best_test_loss:
                            best_test_loss = test_log_loss
                            best_model_state = {
                                k: v.clone() for k, v in model.state_dict().items()
                            }
                            epochs_without_improvement = 0
                        else:
                            epochs_without_improvement += train_args["eval_period"]
                        postfix["best"] = f"{best_test_loss:.4f}"
                        postfix["pat"] = (
                            f"{epochs_without_improvement}"
                            f"/{early_stopping_patience}"
                        )
                pbar.set_postfix(postfix)
                sum_loss = 0
                n_steps = 0

            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                pbar.close()
                print(
                    f"  Early stopping at epoch {e} "
                    f"(best test loss: {best_test_loss:.4f})"
                )
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

        if test_data is not None:
            # compute confusion matrix
            for j, mask in enumerate(test_mask_pattern):
                n_pred = mask.sum().item()
                _d = apply_mask_pattern(
                    test_data, mask[np.newaxis], y_name, mask_name, default_values
                )
                _d = model.encode(
                    _d, mask=mask_name, edge_index=test_edge_index, device=th_device
                )
                conf_m_all.append(
                    create_confusion_matrix(
                        model,
                        _d,
                        y_name=y_name,
                        labels={
                            **labels,
                            "n_pred": n_pred,
                            "mask": j,
                            "set": "test",
                            "cv_split": i,
                        },
                    )
                )

        if i is None:
            model.save(model_path)

        if len(conf_m_all) > 0:
            conf_m = pd.concat(conf_m_all)
            conf_m.to_parquet(conf_path)
        rec.save(output_dir, labels, job_id=job_id)

        final_test_losses.append(last_test_log_loss)

    emit(
        "train.final_test_loss_per_fold",
        [round(v, 4) if v is not None else None for v in final_test_losses],
    )

    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    config_path = sys.argv[1]
    config = load_config(config_path)
    main(config)
