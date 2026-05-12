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

from aimanager.manager.memory import Memory
from aimanager.manager.environment import ArtificialHumanEnv
from aimanager.artificial_humans import AH_MODELS
from aimanager.manager.manager import ArtificalManager
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
]

# Will be set in train_manager based on the encoding config
replay_keys = []


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

    state = env.reset()
    metric_list = []
    for round_number in count():
        statecopy = {k: v.clone() for k, v in state.items() if k in replay_keys}

        action, q_values = manager.get_action(state, greedy=on_policy)

        # Two-manager mode: RL produces (B, 8, 1) over all agents; opponent
        # produces its own (B, 8, 1); mask keeps each manager's own group.
        # agent_groups mutates per round via the switch predictor, so the
        # mask is recomputed every step.
        if opponent_manager is not None:
            opp_action, _ = opponent_manager.predict(
                state,
                reset_rnn=round_number == 0,
                edge_index=env.batch_edge_index,
            )
            rl_mask = (env.agent_groups.squeeze(-1) == rl_group_id).unsqueeze(-1)
            final_punishment = th.where(rl_mask, action, opp_action)
        else:
            final_punishment = action

        state = env.punish(final_punishment)

        metrics = {k: state[k].to(th.float).mean().item() for k in rec_keys}

        # pass actions to environment and advance by one step
        state, reward, done = env.step()
        if replay_mem is not None:
            replay_mem.add(
                episode_step=round_number,
                episode=update_step,
                action=action,
                reward=reward,
                **statecopy,
            )

        metrics["next_reward"] = reward.mean().item()
        metrics["q_min"] = q_values.min().item()
        metrics["q_max"] = q_values.max().item()
        metrics["q_mean"] = q_values.mean().item()
        metrics["round_number"] = round_number
        metrics["sampling"] = "greedy" if on_policy else "eps-greedy"
        metrics["update_step"] = update_step
        metric_list.append(metrics)

        if done:
            break
    return metric_list


def train_manager(config: dict, labels=None, data_dir: str = None):
    """Train the manager using the provided config."""
    global replay_keys

    if labels is None:
        labels = {}

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
    opponent_manager_path = os.path.join(basedir, config["opponent_manager"])
    print(f"Loading opponent manager from {opponent_manager_path}")
    opponent_manager = (
        AH_MODELS[config["artificial_humans_model"]]
        .load(opponent_manager_path, device=device)
        .to(device)
    )
    rl_group_id = config.get("rl_group_id", 0)

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
                for k in [
                    "next_reward",
                    "common_good",
                    "contribution",
                    "punishment",
                    "group_payoff",
                    "q_mean",
                ]:
                    log[f"eval/{k}"] = sum(m[k] for m in on_policy_metrics) / len(
                        on_policy_metrics
                    )
                wandb.log(log)

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
    ]

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
