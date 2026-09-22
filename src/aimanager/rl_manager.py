import json
import os
import sys
import random
import warnings
import yaml

from itertools import count
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch as th
import wandb

from aimanager.evaluation_suite.metrics import RPA_EDGES, RPA_LABELS
from aimanager.manager.memory import Memory
from aimanager.manager.environment import ArtificialHumanEnv
from aimanager.artificial_humans import AH_MODELS
from aimanager.manager import head_probe
from aimanager.manager.manager import BOOTSTRAP, ArtificalManager
from aimanager.manager.linear_opponent import load_opponent
from aimanager.utils.utils import make_dir
from aimanager.utils.array_to_df import add_labels

# pytorch geometric meta module has changed place
# since the the saving of the training data, this points
# to the new location
import torch_geometric.nn.models.meta as meta_module

sys.modules["torch_geometric.nn.meta"] = meta_module


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "rl_manager.yml")

rec_keys = [
    "punishment",
    "contribution",
    "common_good",
    "contributor_payoff",
    # "manager_payoff",
    "group_payoff",
    "group_payoff_sum",
]

# Will be set in train_manager based on the encoding config
replay_keys = []

# The comparison's budget is equal ENVIRONMENT EPISODES, not equal update
# steps, so the consumption is counted rather than argued. Every `run_batch`
# call is one `env.reset()` and therefore `env.batch_size` complete episodes;
# the totals are printed at the end of training on a single greppable line.
EPISODE_BUDGET = {
    "rollouts": 0,
    "episodes": 0,
    "episode_rounds": 0,
    "behaviour_episodes": 0,
    "eval_episodes": 0,
}


def rpa_shape(recorded, groups, rl_group_id, prefix="rpa"):
    """Mean punishment per contribution bin -- the policy SHAPE, recorded live.

    Two of the three finished runs came out with this contingency inverted
    (mean punishment rising with contribution, the opposite of the human
    managers' 4.76 -> 0.27), so the shape is the primary outcome of the
    exploration comparison and not a post-hoc diagnostic. Recording it here,
    per rollout, makes it readable for every seed at every evaluation point
    without waiting for a cross-evaluation simulation -- and separately for
    the behaviour and the evaluation rollout, which is the gap the comparison
    is about.

    The bins are `RPA_EDGES` / `RPA_LABELS` imported from the evaluation
    suite, not redefined here, so these numbers sit on exactly the axis
    `scripts/rl_two_worlds/measure.py` prints. `pd.cut` bins are left-open /
    right-closed and `th.bucketize(..., right=False)` is the same convention.

    Cells where the contributor gave no input are dropped: the suite marks
    their contribution NaN (`convert.load_human`), while the env carries an
    imputed default that would land in the {0} bin.

    Called twice per round with `prefix` "rpa" for the RL manager's own group
    and "rpa_opp" for the opponent's, so the artificial punisher -- this
    project's clone of a human manager -- is measured as a shape column on
    exactly the same rollouts, with no extra run.
    """
    c = recorded["contribution"].squeeze(-1).to(th.float)
    p = recorded["punishment"].squeeze(-1).to(th.float)
    keep = recorded["contribution_valid"].squeeze(-1).to(th.bool)
    if rl_group_id is not None:
        keep = keep & (groups == rl_group_id)
    edges = th.tensor(RPA_EDGES, device=c.device, dtype=c.dtype)
    idx = (th.bucketize(c, edges) - 1)[keep]
    # One bincount pair and two device transfers per call, not one transfer
    # per bin: this runs inside the 4000-step training loop and a `.item()`
    # per bin would be 4.6 million synchronisations over a run.
    n_bins = len(RPA_LABELS)
    n = th.bincount(idx, minlength=n_bins).tolist()
    total = th.bincount(idx, weights=p[keep], minlength=n_bins).tolist()
    out = {}
    for b, label in enumerate(RPA_LABELS):
        out[f"{prefix}_n_{label}"] = float(n[b])
        out[f"{prefix}_mean_{label}"] = total[b] / n[b] if n[b] else float("nan")
    return out


def load_config(path: str = None) -> dict:
    """Load YAML config for the RL manager."""
    print(f"Loading config from {path}")
    config_path = path or DEFAULT_CONFIG_PATH
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_batch(
    manager,
    env,
    replay_mem=None,
    on_policy=True,
    update_step=None,
    opponent_manager=None,
    rl_group_id=0,
):

    # The manager is served by the env exactly as the contribution, switch and
    # punisher models are, so it reads the state through `served_state()`: a
    # player who gave no input contributed nothing, and that is what the game
    # charged and what everyone saw. `env.reset()` / `env.step()` return
    # `self.state`, which still carries the default `update_contribution`
    # writes over a timed-out cell -- the value the recorded output needs and
    # the manager must not be trained on (ArtificialHumanEnv.served_state).
    env.reset()
    # One perturbation per episode, drawn here and fixed for all 24 rounds.
    # A rollout is one batch of 1000 parallel episodes and one replay-memory
    # episode, so "once per episode" is once per rollout: all 1000 share the
    # draw, which is what makes the behaviour a single coherent policy.
    if not on_policy:
        manager.begin_behaviour_episode()
    state = env.served_state()

    # Bootstrapped behaviour: one head per parallel episode, drawn here and
    # held for all 24 rounds -- the coherent alternative policy that replaces
    # per-action dithering. The bootstrap mask is drawn with it, once per
    # episode, and travels into the replay buffer with the episode's
    # transitions so it stays a fixed property of the data. Evaluation
    # rollouts (`on_policy=True`) draw neither: they are the ensemble
    # consensus with every exploration mechanism off.
    bootstrap = (not on_policy) and getattr(manager, "exploration", None) == BOOTSTRAP
    head_kwargs = {}
    head_mask = None
    if bootstrap:
        n_batch = state["agent_group"].shape[0]
        head_kwargs["head"] = manager.draw_heads(n_batch)
        head_mask = manager.draw_masks(n_batch)

    metric_list = []
    for round_number in count():
        statecopy = {k: v.clone() for k, v in state.items() if k in replay_keys}

        agent_group_now = state["agent_group"]
        # `head` is passed only when there is one, so a manager without the
        # bootstrap mechanism is called with exactly the signature it has
        # always been called with.
        action, q_values = manager.get_action(
            state, first=round_number == 0, greedy=on_policy, **head_kwargs
        )

        # Two-manager mode: RL produces (B, 8, 1) over all agents; opponent
        # produces its own (B, 8, 1); mask keeps each manager's own group.
        # agent_groups mutates per round via the switch predictor, so the
        # mask is recomputed every step.
        if opponent_manager is not None:
            # reset_rnn only matters if the opponent has an RNN (non-autoreg
            # variant). The autoreg punishment AH ignores it. We pass it
            # uniformly so the same call site supports both opponents.
            opp_action, _ = opponent_manager.predict(
                state,
                reset_rnn=round_number == 0,
                edge_index=env.batch_edge_index,
            )
            rl_mask = (env.agent_groups.squeeze(-1) == rl_group_id).unsqueeze(-1)
            final_punishment = th.where(rl_mask, action, opp_action)
        else:
            final_punishment = action

        # The env's own state, not the served view: these metrics are the
        # record of what the game charged and paid out, so they keep the
        # imputed value exactly as per_round.parquet does.
        recorded = env.punish(final_punishment)

        metrics = {k: recorded[k].to(th.float).mean().item() for k in rec_keys}

        # Pre-step agent_groups: mask reflects who received this round's
        # punishment. Assumes n_groups == 2.
        if opponent_manager is not None:
            opp_group_id = 1 - rl_group_id
            groups = env.agent_groups.squeeze(-1)
            rl_mask = (groups == rl_group_id).float()
            opp_mask = (groups == opp_group_id).float()
            rl_count = rl_mask.sum(dim=1).clamp(min=1)
            opp_count = opp_mask.sum(dim=1).clamp(min=1)
            for k in (
                "punishment",
                "contribution",
                "common_good",
                "contributor_payoff",
            ):
                x = recorded[k].squeeze(-1).to(th.float)
                metrics[k] = ((x * rl_mask).sum(dim=1) / rl_count).mean().item()
            for k in ("group_payoff", "group_payoff_sum"):
                metrics[k] = recorded[k][:, rl_group_id].to(th.float).mean().item()
            opp_p = recorded["punishment"].squeeze(-1).to(th.float)
            metrics["opp_punishment"] = (
                ((opp_p * opp_mask).sum(dim=1) / opp_count).mean().item()
            )
            metrics["opp_sum_payoff"] = (
                recorded["group_payoff_sum"][:, opp_group_id].to(th.float).mean().item()
            )

        # Must be read BEFORE `env.step()`: `recorded` is `env.state` itself
        # and `step()` overwrites the contribution in place.
        if opponent_manager is not None:
            metrics.update(rpa_shape(recorded, groups, rl_group_id))
            metrics.update(
                rpa_shape(recorded, groups, 1 - rl_group_id, prefix="rpa_opp")
            )
        else:
            metrics.update(rpa_shape(recorded, None, None))

        # pass actions to environment and advance by one step
        _, reward, done = env.step()
        state = env.served_state()
        if replay_mem is not None:
            extra = {} if head_mask is None else {"head_mask": head_mask}
            replay_mem.add(
                episode_step=round_number,
                episode=update_step,
                action=action,
                reward=reward,
                **extra,
                **statecopy,
            )

        if opponent_manager is not None:
            metrics["next_reward"] = (
                reward[:, rl_group_id].to(th.float).mean().item()
            )
        else:
            metrics["next_reward"] = reward.mean().item()
        metrics["q_min"] = q_values.min().item()
        metrics["q_max"] = q_values.max().item()
        metrics["q_mean"] = q_values.mean().item()
        if opponent_manager is not None:
            groups_after = env.agent_groups.squeeze(-1)
            rl_size = (
                (groups_after == rl_group_id).float().sum(dim=1).mean().item()
            )
            opp_size = (
                (groups_after != rl_group_id).float().sum(dim=1).mean().item()
            )
            # Same per-round value; aggregator (last vs mean across rounds)
            # differs downstream: end_* keys are taken at the final round,
            # avg_* keys are averaged across the episode.
            metrics["rl_end_group_size"] = rl_size
            metrics["rl_avg_group_size"] = rl_size
            metrics["opp_end_group_size"] = opp_size
            metrics["opp_avg_group_size"] = opp_size
        # The policy's shape -- mean punishment binned by the contribution it
        # was aimed at, on the evaluation suite's own RPA bins -- and, when
        # there is an ensemble, what each head would have done on the same
        # cells. Restricted to the RL manager's own group: the opponent's
        # punishments are not this policy.
        rl_cells = (
            (agent_group_now == rl_group_id)
            if opponent_manager is not None
            else th.ones_like(agent_group_now, dtype=th.bool)
        )
        metrics.update(
            head_probe.shape_metrics(
                recorded["punishment"], recorded["contribution"], rl_cells
            )
        )
        if getattr(manager, "n_heads", 1) > 1:
            head_acts = head_probe.gather_to_own_group(
                q_values.argmax(-1), agent_group_now
            )
            metrics.update(
                head_probe.head_metrics(
                    q_values,
                    head_acts,
                    recorded["contribution"],
                    rl_cells,
                    agent_group_now,
                )
            )

        metrics["round_number"] = round_number
        # Three exploration arms share this column. `behaviour_label` covers
        # the two that live on the manager (param-noise / eps-greedy);
        # bootstrap is decided per rollout by the head draw, so it is named
        # here.
        metrics["sampling"] = (
            "greedy"
            if on_policy
            else ("bootstrap-head" if bootstrap else manager.behaviour_label)
        )
        metrics["update_step"] = update_step
        metric_list.append(metrics)

        if done:
            break

    EPISODE_BUDGET["rollouts"] += 1
    EPISODE_BUDGET["episodes"] += env.batch_size
    EPISODE_BUDGET["episode_rounds"] += env.batch_size * len(metric_list)
    key = "eval_episodes" if on_policy else "behaviour_episodes"
    EPISODE_BUDGET[key] += env.batch_size
    if not on_policy:
        # Adapts the scale for the NEXT episode and reports the scale this one
        # was actually collected under, stamped on every round of it.
        noise_stats = manager.end_behaviour_episode()
        for m in metric_list:
            m.update(noise_stats)
    return metric_list


def train_manager(config: dict, labels=None, data_dir: str = None):
    """Train the manager using the provided config."""
    global replay_keys

    if labels is None:
        labels = {}

    for k in EPISODE_BUDGET:
        EPISODE_BUDGET[k] = 0

    device = th.device(config["device"])
    cpu = th.device("cpu")

    # Seeding
    print(f"Seeding with seed {config['seed']}")
    seed = config["seed"]
    th.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    wandb_enabled = bool(os.environ.get("WANDB_API_KEY"))
    if wandb_enabled:
        wandb.init(config=config)

    # Output directories
    output_dir = data_dir or config["output_dir"]
    metrics_dir = os.path.join(output_dir, "metrics")
    model_dir = os.path.join(output_dir, "model")
    print(f"Output directories: metrics_dir={metrics_dir}," f" model_dir={model_dir}")
    make_dir(metrics_dir)
    make_dir(model_dir)

    # Model paths
    basedir = config["basedir"]
    artificial_humans_path = os.path.join(basedir, config["artificial_humans"])
    artificial_humans_valid_path = os.path.join(
        basedir, config["artificial_humans_valid"]
    )

    print(
        f"Loading artificial humans from {artificial_humans_path}"
        f" and {artificial_humans_valid_path}"
    )
    ah = (
        AH_MODELS[config["artificial_humans_model"]]
        .load(artificial_humans_path, device=device)
        .to(device)
    )
    ahv = (
        AH_MODELS[config["artificial_humans_model"]]
        .load(artificial_humans_valid_path, device=device)
        .to(device)
    )

    # Fixed opponent that controls the non-RL group. Generic key so a later
    # self-play setup can swap in an RL-manager checkpoint without renaming.
    # Optional for backwards compatibility with legacy single-group configs
    # (e.g. configs/training/rl_manager/02_rnn_node_1group.yml); when absent
    # the rollout runs single-manager and the TD-error covers all groups.
    opponent_manager = None
    if "opponent_manager" in config:
        opponent_manager_path = os.path.join(basedir, config["opponent_manager"])
        print(f"Loading opponent manager from {opponent_manager_path}")
        # `.joblib` -> the batched linear punisher, anything else -> a GNN
        # punisher: the same extension dispatch the simulation configs use, so
        # this slot can name a linear baseline where a GNN artifact used to sit.
        opponent_manager = load_opponent(
            opponent_manager_path,
            n_groups=config["env_args"].get("n_groups", 1),
            device=device,
        ).to(device)

    # Switch predictor — required for group-switching dynamics. Optional
    # for backwards compatibility with legacy single-group configs. Key name
    # mirrors configs/simulation/ah_testing/group_switching_ah_punishment_50ep.yml.
    switch_model = None
    if "switch_model" in config:
        switch_model_path = os.path.join(basedir, config["switch_model"])
        print(f"Loading switch predictor from {switch_model_path}")
        switch_model = (
            AH_MODELS[config["artificial_humans_model"]]
            .load(switch_model_path, device=device)
            .to(device)
        )

    env_args = config["env_args"].copy()
    # rl_group_id sits under env_args for organisation (it's a property of
    # the multi-group setup) but is consumed by the training loop, not by
    # the env constructor — pop it before the env build.
    rl_group_id = env_args.pop("rl_group_id", 0)
    if env_args.pop("reward_formula", None) is not None:
        warnings.warn(
            "reward_formula is deprecated and ignored. "
            "Reward is always computed from group_payoff.",
            DeprecationWarning,
            stacklevel=2,
        )

    print(f"Creating environment with {env_args}")
    env = ArtificialHumanEnv(
        artifical_humans=ah,
        artifical_humans_valid=ahv,
        artifical_humans_switch=switch_model,
        device=device,
        **env_args,
    )

    manager_args = config["manager_args"]
    manager = ArtificalManager(
        n_contributions=env.n_contributions,
        n_punishments=env.n_punishments,
        n_groups=env.n_groups,
        default_values=ah.default_values,
        device=device,
        **manager_args,
    )

    replay_mem = Memory(
        n_episode_steps=env.n_rounds,
        device=cpu,
        **config["replay_memory_args"],
    )

    # derive replay_keys from encoding config
    model_args = manager_args["model_args"]
    replay_keys = [n["name"] for n in model_args["x_encoding"]]
    replay_keys += [n["name"] for n in model_args["b_encoding"]]
    replay_keys += ["punishment"]
    replay_keys += ["agent_group"]
    replay_keys = list(set(replay_keys))

    metrics_list = []

    n_update_steps = config["n_update_steps"]
    training_batch_size = config["training_batch_size"]
    eval_period = config["eval_period"]

    print(f"Training manager for {n_update_steps} update steps")
    if opponent_manager is not None:
        print(
            f"[two-manager] rl_group_id={rl_group_id}, "
            f"env.n_groups={env.n_groups}, env.n_agents={env.n_agents}, "
            f"reward_mode={env.reward_mode}, "
            f"switch_predictor={'on' if switch_model is not None else 'off'}"
        )
    for update_step in tqdm(range(n_update_steps)):
        # here we sample one batch of episodes and add them to the replay buffer
        off_policy_metrics = run_batch(
            manager,
            env,
            replay_mem,
            on_policy=False,
            update_step=update_step,
            opponent_manager=opponent_manager,
            rl_group_id=rl_group_id,
        )

        replay_mem.next_episode(update_step)

        # allow manager to update itself
        sample = replay_mem.get_random(device=device, n_episodes=training_batch_size)

        if sample is not None:
            loss = manager.update(
                update_step,
                **sample,
                batch=env.batch,
                edge_index=env.batch_edge_index,
                agent_group_mask=env.agent_group_mask,
                rl_group_id=rl_group_id if opponent_manager is not None else None,
            )

        if (update_step % eval_period) == 0:
            if sample is not None:
                avg_reward = sum(m["next_reward"] for m in off_policy_metrics) / len(
                    off_policy_metrics
                )
                print(
                    f"Step {update_step} |"
                    f" Loss {loss.item():.4f} |"
                    f" Reward {avg_reward:.4f}"
                )
                metrics_list.extend(
                    [{**m, "loss": loss.item()} for m in off_policy_metrics]
                )
            on_policy_metrics = run_batch(
                manager,
                env,
                replay_mem=None,
                on_policy=True,
                update_step=update_step,
                opponent_manager=opponent_manager,
                rl_group_id=rl_group_id,
            )
            metrics_list.extend(on_policy_metrics)

            if wandb_enabled:
                log = {"update_step": update_step}
                if sample is not None:
                    log["train/loss"] = loss.item()
                eval_keys = [
                    "next_reward",
                    "common_good",
                    "contribution",
                    "punishment",
                    "group_payoff",
                    "group_payoff_sum",
                    "q_mean",
                ]
                for k in eval_keys:
                    log[f"eval/{k}"] = sum(m[k] for m in on_policy_metrics) / len(
                        on_policy_metrics
                    )
                if opponent_manager is not None:
                    last = on_policy_metrics[-1]
                    log["eval/rl_end_group_size"] = last["rl_end_group_size"]
                    log["eval/opp_end_group_size"] = last["opp_end_group_size"]
                    for k in (
                        "rl_avg_group_size",
                        "opp_avg_group_size",
                        "opp_punishment",
                        "opp_sum_payoff",
                    ):
                        log[f"eval/{k}"] = sum(
                            m[k] for m in on_policy_metrics
                        ) / len(on_policy_metrics)
                wandb.log(log)

    print(f"[budget] {json.dumps(EPISODE_BUDGET, sort_keys=True)}")

    model_file = os.path.join(model_dir, f"{config['job_id']}_manager.pt")
    print(f"Saving manager to {model_file}")

    manager.save(model_file)

    # test model saving and loading
    print(f"Loading manager from {model_file}")
    manager.load(model_file, device=device)

    id_vars = ["round_number", "sampling", "update_step"]
    value_vars = [
        "punishment",
        "contribution",
        "common_good",
        "contributor_payoff",
        # "manager_payoff",
        "next_reward",
        "q_min",
        "q_max",
        "q_mean",
        "loss",
        "group_payoff",
        "group_payoff_sum",
    ]
    # Two-manager-only metrics; melt only when recorded so legacy
    # single-manager parquets are unchanged.
    if metrics_list and "rl_end_group_size" in metrics_list[0]:
        value_vars.extend(
            k
            for k in metrics_list[0]
            if k.startswith("rl_") or k.startswith("opp_")
        )
    # The policy-shape rows and each exploration arm's own diagnostics, melted
    # under their own names so the metric names the comparison shares (above)
    # keep their exact meaning -- never a rename of a contract metric into
    # another arm's slot. The union over records, not the first record: only
    # behaviour rollouts carry the noise keys, only two-manager runs carry the
    # shape keys, and only the ensemble arm carries the head keys.
    seen = set(value_vars)
    for row in metrics_list:
        for k in row:
            prefixes = ("rpa_", "head_", "consensus_", "param_noise_")
            if k.startswith(prefixes) and k not in seen:
                seen.add(k)
                value_vars.append(k)

    metrics_path = os.path.join(metrics_dir, f"{config['job_id']}.parquet")
    print(f"Saving metrics dataframe to {metrics_path}")
    df = pd.DataFrame.from_records(metrics_list)
    df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="metric")
    df = add_labels(df, {**labels, "job_id": config["job_id"]})
    df.to_parquet(metrics_path)

    if wandb_enabled:
        wandb.finish()

    return model_file


def main(config):
    """Entry point for the unified CLI."""
    train_manager(config)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    config = load_config(config_path)
    main(config)
