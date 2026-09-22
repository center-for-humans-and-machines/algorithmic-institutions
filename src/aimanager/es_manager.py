"""Evolution strategies for the RL manager (Salimans, Ho, Chen, Sidor and
Sutskever, 2017, "Evolution Strategies as a Scalable Alternative to
Reinforcement Learning").

Why this exists
---------------
The DQN manager explores with epsilon-greedy fixed at 0.1 over 31 ordinal
punishment levels. A uniform draw over 0..30 has mean 15, so eps=0.1 injects
1.5 punishment points per member per round -- the size of the entire learned
signal -- and it injects them *independently of the contribution they are
aimed at*. Two things follow that this module removes by construction:

  * the policy being evaluated is never the policy being run, so each run
    scores a behaviour policy that punishes 1.7 to 6.6 times as hard as its
    own greedy policy;
  * punishment is decorrelated from contribution in the replay buffer, which
    is a candidate route to the inverted policy shape measured on the three
    finished DQN seeds (two of three punish the full contributor HARDER than
    the free rider).

Here every policy that is ever scored is a fixed deterministic policy, run
over complete episodes, and judged on the whole-episode consequences of the
contribution -> punishment mapping it implements. There is no action noise at
any point.

Method
------
Mirrored (antithetic) sampling of a population of parameter perturbations,
centered-rank fitness shaping, Adam on the fitness-weighted average. All three
are the paper's own defaults and all three are load-bearing here:

  * MIRRORED SAMPLING halves the number of independent directions but turns
    each pair into a two-sided finite difference, which cancels the fitness
    level of theta itself. Without it the estimator carries that level as a
    bias term weighted by the mean perturbation, and with a population as
    small as ours the mean perturbation is not small.
  * CENTERED-RANK SHAPING maps the P fitnesses onto a fixed grid in
    [-0.5, 0.5]. The episode return here has a heavy component from group
    size (the switch predictor moves members between groups, so a lucky
    member can be scored on a group of 8 and an unlucky one on a group of 0)
    and raw-return weighting would let one such outlier set the whole update
    direction. Rank shaping bounds every member's influence. Skipping it is
    the single most common reason ES underperforms and it is not skipped.
  * ADAM on the estimated gradient, with the paper's L2 coefficient, because
    the ES estimator's scale drifts as the fitness spread changes and a plain
    SGD step would change effective size with it.

Batched population evaluation
-----------------------------
The population is evaluated inside ONE env rollout. `batch_size` episodes are
partitioned along the batch dimension, member p owning the contiguous slice
[p*E, (p+1)*E). Each member is a full `GraphNetwork` clone with its own
recurrent state, so the manager head runs P times per round on 1/P of the
batch; the expensive part of the environment -- the contribution, validity and
switch artificial humans and the opponent punisher -- runs ONCE over the whole
batch, exactly as it does for the DQN arm. See the module docstring of
`scripts/rl_es/pilot.py` for the measured cost of that choice.

Budget and axes
---------------
One generation consumes exactly `batch_size` behaviour episodes, the same
number one DQN update step consumes. The generation index is written into the
`update_step` column of the metrics parquet SO THAT THE AXES LINE UP: for both
arms, `update_step = (behaviour episodes so far) / batch_size`. It is NOT an
update step in the DQN sense -- this method has no gradient steps on a replay
buffer -- and any reader comparing arms must read it as an episode counter.
"""

import copy
import os
import random
import sys
import time
from itertools import count

import numpy as np
import pandas as pd
import torch as th
import wandb

from aimanager.artificial_humans import AH_MODELS
from aimanager.evaluation_suite.metrics import RPA_EDGES, RPA_LABELS
from aimanager.manager.environment import ArtificialHumanEnv
from aimanager.manager.linear_opponent import load_opponent
from aimanager.manager.manager import ArtificalManager
from aimanager.utils.array_to_df import add_labels
from aimanager.utils.utils import make_dir

# Same relocation the DQN path applies before unpickling the artifacts.
import torch_geometric.nn.models.meta as meta_module  # noqa: E402

sys.modules["torch_geometric.nn.meta"] = meta_module


#: Behaviour rollouts: the population members, each a deterministic policy.
SAMPLING_BEHAVIOUR = "es-population"
#: Evaluation rollouts: the mean parameter vector, deterministic. Same tag as
#: the DQN arm's evaluation rollout, so a cross-arm comparison is one filter.
SAMPLING_EVAL = "greedy"

#: The contribution bins of the policy-shape read, taken from the evaluation
#: suite rather than restated, so this arm and
#: plots/data_analysis/evaluation/rl_manager_two_worlds/policy_shape.csv bin
#: identically. `pd.cut` is right-closed, so bin index = the number of
#: interior edges the contribution exceeds.
RPA_THRESHOLDS = tuple(RPA_EDGES[1:-1])
N_RPA_BINS = len(RPA_LABELS)

#: Metric keys the ES arm records. The DQN arm's `q_min` / `q_max` / `q_mean`
#: are deliberately absent: this method has no Q values and renaming something
#: else into their slots would be a lie in a column a reader will trust.
VALUE_VARS = [
    "punishment",
    "opp_punishment",
    "rl_avg_group_size",
    "opp_avg_group_size",
    "rl_end_group_size",
    "opp_end_group_size",
    "contribution",
    "common_good",
    "contributor_payoff",
    "group_payoff",
    "group_payoff_sum",
    "opp_sum_payoff",
    "next_reward",
]


# -- parameter vector plumbing ---------------------------------------


def flat_params(model):
    """The model's parameters as one contiguous 1-D vector."""
    return th.cat([p.data.reshape(-1) for p in model.parameters()])


def set_flat_params_(model, vec):
    """Write a flat vector back into the model's parameters, in place."""
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(vec[i : i + n].view_as(p))
        i += n
    assert i == vec.numel(), f"parameter vector size mismatch: {i} != {vec.numel()}"


def centered_ranks(fitness):
    """Map fitnesses onto an evenly spaced grid in [-0.5, 0.5] by rank.

    Scale-free and outlier-bounded: only the ORDER of the members survives, so
    a member that happened to be scored on an unusually large group cannot
    dominate the update through the size of its return.
    """
    n = fitness.numel()
    order = fitness.argsort()
    ranks = th.empty_like(fitness)
    ranks[order] = th.arange(n, dtype=fitness.dtype, device=fitness.device)
    if n == 1:
        return th.zeros_like(fitness)
    return ranks / (n - 1) - 0.5


def draw_perturbations(n_members, n_params, generator, device, mirrored=True):
    """Population of perturbations, (P, D).

    With `mirrored`, member 2k+1 carries exactly the negation of member 2k, so
    each adjacent pair is a two-sided finite difference along one direction.
    """
    if mirrored:
        assert n_members % 2 == 0, "mirrored sampling needs an even population"
        half = th.randn((n_members // 2, n_params), generator=generator, device=device)
        return th.stack([half, -half], dim=1).reshape(n_members, n_params)
    return th.randn((n_members, n_params), generator=generator, device=device)


# -- policy shape ----------------------------------------------------


def rpa_bin(contribution):
    """Contribution -> evaluation-suite RPA bin index, 0..5."""
    idx = th.zeros_like(contribution, dtype=th.int64)
    for edge in RPA_THRESHOLDS:
        idx = idx + (contribution > edge).to(th.int64)
    return idx


class ShapeAccumulator:
    """Mean realised punishment per (member, contribution bin).

    Mirrors what `evaluation_suite.convert.load_sim` + `ResponseMetrics.rpa`
    would compute off a `per_round.parquet` of this rollout, restricted to the
    RL manager's own group: the RAW env contribution (carrying the imputed
    default on a timed-out cell, exactly as the recorded output does) against
    the REALISED punishment (zeroed on a timed-out cell by `env.punish`).

    `valid` additionally keeps only the cells where the player really gave an
    input. The human frame marks those cells NaN and drops them; the
    simulation frame does not, so the two subsets bracket the comparison and
    the log reports both rather than picking one.
    """

    def __init__(self, n_members, device):
        self.n_members = n_members
        shape = (n_members, N_RPA_BINS)
        self.sum = th.zeros(shape, dtype=th.float64, device=device)
        self.n = th.zeros(shape, dtype=th.float64, device=device)
        self.sum_valid = th.zeros(shape, dtype=th.float64, device=device)
        self.n_valid = th.zeros(shape, dtype=th.float64, device=device)

    def add(self, env, rl_group_id, member_of_episode):
        in_rl = (env.agent_groups.squeeze(-1) == rl_group_id).reshape(-1)
        contribution = env.contribution.squeeze(-1).reshape(-1)
        punishment = env.punishment.squeeze(-1).reshape(-1).to(th.float64)
        valid = env.contribution_valid.squeeze(-1).reshape(-1)
        member = member_of_episode.unsqueeze(-1).expand(-1, env.n_agents).reshape(-1)
        cell = member * N_RPA_BINS + rpa_bin(contribution)
        for mask, acc_sum, acc_n in (
            (in_rl, self.sum, self.n),
            (in_rl & valid, self.sum_valid, self.n_valid),
        ):
            acc_sum.view(-1).index_add_(0, cell[mask], punishment[mask])
            acc_n.view(-1).index_add_(0, cell[mask], th.ones_like(punishment[mask]))

    def rows(self, sampling, update_step, per_member):
        """Long-format rows; `member` is -1 for the pooled read."""
        out = []
        pooled = [
            (
                -1,
                self.sum.sum(0),
                self.n.sum(0),
                self.sum_valid.sum(0),
                self.n_valid.sum(0),
            )
        ]
        members = (
            [
                (m, self.sum[m], self.n[m], self.sum_valid[m], self.n_valid[m])
                for m in range(self.n_members)
            ]
            if per_member
            else []
        )
        for member, s, n, sv, nv in pooled + members:
            for subset, num, den in (("all", s, n), ("valid", sv, nv)):
                for b, label in enumerate(RPA_LABELS):
                    count_ = float(den[b])
                    out.append(
                        {
                            "update_step": update_step,
                            "sampling": sampling,
                            "member": member,
                            "subset": subset,
                            "contribution_bin": label,
                            "mean_punishment": (
                                float(num[b]) / count_ if count_ else float("nan")
                            ),
                            "n": count_,
                        }
                    )
        return out


# -- rollout ---------------------------------------------------------


def round_metrics(env, recorded, reward, rl_group_id):
    """One round's metrics, in the DQN arm's own definitions.

    Deliberately a transcription of `rl_manager.run_batch`'s two-manager
    block rather than an import: `run_batch` interleaves the metric
    computation with epsilon-greedy action selection and a replay write, and
    this arm has neither. `test_es_rollout_matches_run_batch` pins the two
    together numerically so the transcription cannot drift silently.
    """
    metrics = {}
    opp_group_id = 1 - rl_group_id
    groups = env.agent_groups.squeeze(-1)
    rl_mask = (groups == rl_group_id).float()
    opp_mask = (groups == opp_group_id).float()
    rl_count = rl_mask.sum(dim=1).clamp(min=1)
    opp_count = opp_mask.sum(dim=1).clamp(min=1)
    for k in ("punishment", "contribution", "common_good", "contributor_payoff"):
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
    metrics["next_reward"] = reward[:, rl_group_id].to(th.float).mean().item()
    return metrics


def group_size_metrics(env, rl_group_id):
    groups_after = env.agent_groups.squeeze(-1)
    rl_size = (groups_after == rl_group_id).float().sum(dim=1).mean().item()
    opp_size = (groups_after != rl_group_id).float().sum(dim=1).mean().item()
    return {
        "rl_end_group_size": rl_size,
        "rl_avg_group_size": rl_size,
        "opp_end_group_size": opp_size,
        "opp_avg_group_size": opp_size,
    }


class PopulationRollout:
    """One env rollout that scores `len(members)` deterministic policies.

    `members` are `ArtificalManager`s over clones of the same architecture;
    member p acts on episodes [p*E, (p+1)*E) of the shared batch. With a
    single member this is the deterministic evaluation rollout and is
    numerically identical to `rl_manager.run_batch(..., on_policy=True)`.
    """

    def __init__(self, env, opponent_manager, rl_group_id, device):
        self.env = env
        self.opponent_manager = opponent_manager
        self.rl_group_id = rl_group_id
        self.device = device
        self._edge_index_cache = {}

    def _edge_index(self, model, n_player, n_batch):
        # `GraphNetwork.encode` rebuilds this in Python on every call
        # (112,000 index pairs for the DQN arm's full batch). It depends only
        # on the shape, and this arm calls the head P times per round, so it
        # is built once per shape and reused. Values are whatever
        # `create_fully_connected` returns, so nothing about the forward pass
        # changes.
        key = (n_player, n_batch)
        if key not in self._edge_index_cache:
            self._edge_index_cache[key] = model.create_fully_connected(
                n_player, n_batch=n_batch
            )
        return self._edge_index_cache[key]

    def _population_action(self, members, state, first):
        batch_size = self.env.batch_size
        n_members = len(members)
        chunk = batch_size // n_members
        actions = []
        for p, member in enumerate(members):
            sl = slice(p * chunk, (p + 1) * chunk)
            sub = {
                k: (v[sl] if v.shape[0] == batch_size else v) for k, v in state.items()
            }
            edge_index = self._edge_index(
                member.policy_model, self.env.n_agents, chunk * member.n_groups
            )
            action, _ = member.get_action(
                sub, first=first, edge_index=edge_index, greedy=True
            )
            actions.append(action)
        return th.cat(actions, dim=0)

    def run(self, members, update_step, sampling, collect_shape=False):
        """Returns (metric rows, per-episode return, shape accumulator)."""
        env = self.env
        n_members = len(members)
        assert env.batch_size % n_members == 0, (
            f"batch_size {env.batch_size} must be divisible by the population "
            f"size {n_members} for the batch partition to be exact"
        )
        chunk = env.batch_size // n_members
        member_of_episode = th.div(
            th.arange(env.batch_size, device=self.device), chunk, rounding_mode="floor"
        )
        shape = ShapeAccumulator(n_members, self.device) if collect_shape else None

        env.reset()
        state = env.served_state()
        episode_return = th.zeros(env.batch_size, device=self.device)
        metric_list = []
        for round_number in count():
            action = self._population_action(members, state, first=round_number == 0)
            opp_action, _ = self.opponent_manager.predict(
                state, reset_rnn=round_number == 0, edge_index=env.batch_edge_index
            )
            rl_mask = (env.agent_groups.squeeze(-1) == self.rl_group_id).unsqueeze(-1)
            recorded = env.punish(th.where(rl_mask, action, opp_action))

            metrics = round_metrics(env, recorded, env.reward, self.rl_group_id)
            if shape is not None:
                shape.add(env, self.rl_group_id, member_of_episode)

            _, reward, done = env.step()
            state = env.served_state()
            episode_return += reward[:, self.rl_group_id, 0]

            metrics.update(group_size_metrics(env, self.rl_group_id))
            metrics["round_number"] = round_number
            metrics["sampling"] = sampling
            metrics["update_step"] = update_step
            metric_list.append(metrics)
            if done:
                break
        return metric_list, episode_return, shape


# -- training --------------------------------------------------------


def build_members(base_model, manager_kwargs, n_members, device):
    """`n_members` independent clones of the policy model, each an
    `ArtificalManager` so the action path is byte-for-byte the DQN arm's."""
    members = []
    for _ in range(n_members):
        clone = copy.deepcopy(base_model).to(device)
        clone.eval()
        members.append(
            ArtificalManager(policy_model=clone, device=device, **manager_kwargs)
        )
    return members


def build_world(config, device, env_overrides=None):
    """Env, opponent and the untrained manager, in the DQN arm's own order.

    The order matters and is copied from `rl_manager.train_manager`: the
    environment's constructor draws the first round's contributions, so it
    consumes the RNG before the manager's networks are built. Building the
    same objects in the same order after the same seed is what makes this
    arm's initial parameter vector identical to the DQN arm's for the same
    seed, which is what makes the five-seed comparison paired rather than
    merely matched. `scripts/rl_es/pilot.py` guard 1 checks it.

    The manager is built with the DQN arm's FULL argument set -- optimiser,
    target network, epsilon and all -- although none of it is used here, for
    the same reason: `ArtificalManager.__init__` constructs the policy network
    and then the target network, and dropping the second would change nothing
    about the first but would leave the claim resting on that fact rather than
    on the code being identical.
    """
    basedir = config["basedir"]
    kind = AH_MODELS[config["artificial_humans_model"]]

    def load(key):
        return kind.load(os.path.join(basedir, config[key]), device=device).to(device)

    ah = load("artificial_humans")
    ahv = load("artificial_humans_valid")
    opponent_manager = load_opponent(
        os.path.join(basedir, config["opponent_manager"]),
        n_groups=config["env_args"].get("n_groups", 1),
        device=device,
    ).to(device)
    switch_model = load("switch_model")

    env_args = dict(config["env_args"])
    rl_group_id = env_args.pop("rl_group_id", 0)
    env_args.update(env_overrides or {})
    env = ArtificialHumanEnv(
        artifical_humans=ah,
        artifical_humans_valid=ahv,
        artifical_humans_switch=switch_model,
        device=device,
        **env_args,
    )

    manager_args = config["manager_args"]
    base = ArtificalManager(
        n_contributions=env.n_contributions,
        n_punishments=env.n_punishments,
        n_groups=env.n_groups,
        default_values=ah.default_values,
        device=device,
        model_args=manager_args["model_args"],
        opt_args=manager_args["opt_args"],
        gamma=manager_args["gamma"],
        target_update_freq=manager_args["target_update_freq"],
        eps=manager_args["eps"],
    )
    base.policy_model.eval()
    return env, opponent_manager, base, rl_group_id


def member_kwargs_for(env, base):
    return dict(
        n_contributions=env.n_contributions,
        n_punishments=env.n_punishments,
        n_groups=env.n_groups,
        default_values=base.default_values,
    )


def train_manager_es(config, labels=None, data_dir=None):
    """Train the manager with evolution strategies."""
    if labels is None:
        labels = {}

    device = th.device(config["device"])
    seed = config["seed"]
    print(f"Seeding with seed {seed}")
    th.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    wandb_enabled = bool(os.environ.get("WANDB_API_KEY"))
    if wandb_enabled:
        wandb.init(config=config)

    output_dir = data_dir or config["output_dir"]
    metrics_dir = os.path.join(output_dir, "metrics")
    model_dir = os.path.join(output_dir, "model")
    make_dir(metrics_dir)
    make_dir(model_dir)

    env, opponent_manager, base, rl_group_id = build_world(config, device)
    member_kwargs = member_kwargs_for(env, base)

    es = config["es_args"]
    n_members = es["population_size"]
    sigma = float(es["sigma"])
    lr = float(es["lr"])
    l2_coeff = float(es.get("l2_coeff", 0.0))
    mirrored = bool(es.get("mirrored", True))
    shaping = es.get("fitness_shaping", "centered_rank")
    assert shaping in ("centered_rank", "raw"), shaping
    n_generations = config["n_generations"]
    eval_period = config["eval_period"]

    members = build_members(base.policy_model, member_kwargs, n_members, device)
    theta = flat_params(base.policy_model).clone()
    n_params = theta.numel()
    optimizer = th.optim.Adam([theta.requires_grad_(True)], lr=lr)

    # The perturbation stream is its own generator, so it does not interleave
    # with the artificial humans' sampling: the noise a generation sees is a
    # function of the seed and the generation index alone, and stays so if the
    # env's RNG consumption ever changes.
    noise_gen = th.Generator(device=device)
    noise_gen.manual_seed(seed + 10007)

    rollout = PopulationRollout(env, opponent_manager, rl_group_id, device)

    episodes_per_generation = env.batch_size
    print(
        "[ES] Salimans et al. 2017: mirrored sampling, "
        f"{shaping} fitness shaping, Adam.\n"
        f"[ES] population={n_members}, episodes/member={env.batch_size // n_members},"
        f" sigma={sigma}, lr={lr}, l2={l2_coeff}, params={n_params}\n"
        f"[ES] generations={n_generations}, eval every {eval_period} generations\n"
        f"[ES] AXIS NOTE: `update_step` in the metrics parquet is the"
        f" GENERATION index, not a gradient step. One generation ="
        f" {episodes_per_generation} behaviour episodes, the same as one DQN"
        f" update step, so `update_step` means the same number of environment"
        f" episodes in both arms and nothing else.\n"
        f"[ES] behaviour episode budget ="
        f" {n_generations * episodes_per_generation}"
    )

    metrics_list, shape_rows, gen_rows = [], [], []
    for generation in range(n_generations):
        t0 = time.time()
        eps = draw_perturbations(n_members, n_params, noise_gen, device, mirrored)
        with th.no_grad():
            for p, member in enumerate(members):
                set_flat_params_(member.policy_model, theta + sigma * eps[p])

        collect = (generation % eval_period) == 0
        behaviour_metrics, episode_return, shape = rollout.run(
            members,
            update_step=generation,
            sampling=SAMPLING_BEHAVIOUR,
            collect_shape=collect,
        )
        fitness = episode_return.reshape(n_members, -1).mean(dim=1)
        within_se = (
            episode_return.reshape(n_members, -1).std(dim=1).mean()
            / (env.batch_size / n_members) ** 0.5
        )

        weights = centered_ranks(fitness) if shaping == "centered_rank" else fitness
        with th.no_grad():
            grad = (weights.unsqueeze(1) * eps).sum(dim=0) / (n_members * sigma)
            # Ascent on fitness, so the optimiser's descent direction is -g.
            # The L2 term is the paper's, applied to the objective rather than
            # to the step, which keeps |theta| from growing and quietly
            # shrinking the effective noise scale.
            theta.grad = -(grad - l2_coeff * theta)
        optimizer.step()

        wall = time.time() - t0
        gen_row = {
            "generation": generation,
            "behaviour_episodes": (generation + 1) * episodes_per_generation,
            "fitness_mean": float(fitness.mean()),
            "fitness_std": float(fitness.std()),
            "fitness_min": float(fitness.min()),
            "fitness_max": float(fitness.max()),
            "within_member_se": float(within_se),
            "grad_norm": float(grad.norm()),
            "theta_norm": float(theta.detach().norm()),
            "behaviour_punishment": float(
                np.mean([m["punishment"] for m in behaviour_metrics])
            ),
            "wall_clock_s": wall,
        }

        if collect:
            with th.no_grad():
                set_flat_params_(base.policy_model, theta.detach())
            eval_metrics, _, eval_shape = rollout.run(
                [base],
                update_step=generation,
                sampling=SAMPLING_EVAL,
                collect_shape=True,
            )
            metrics_list.extend(behaviour_metrics)
            metrics_list.extend(eval_metrics)
            shape_rows.extend(
                shape.rows(SAMPLING_BEHAVIOUR, generation, per_member=True)
            )
            shape_rows.extend(
                eval_shape.rows(SAMPLING_EVAL, generation, per_member=False)
            )
            eval_punishment = float(np.mean([m["punishment"] for m in eval_metrics]))
            eval_reward = float(np.mean([m["next_reward"] for m in eval_metrics]))
            gen_row["eval_punishment"] = eval_punishment
            gen_row["eval_next_reward"] = eval_reward
            # The headline of this arm: how far the policies that were RUN sit
            # from the policy that was SCORED. Zero action noise means this
            # can only be the perturbation, not the exploration rule.
            gen_row["behaviour_eval_punishment_gap"] = (
                gen_row["behaviour_punishment"] - eval_punishment
            )
            print(
                f"Gen {generation} |"
                f" fitness {gen_row['fitness_mean']:.2f}"
                f" +/-{gen_row['fitness_std']:.2f}"
                f" (within-member se {gen_row['within_member_se']:.2f}) |"
                f" eval reward {eval_reward:.3f} |"
                f" punishment behaviour {gen_row['behaviour_punishment']:.3f}"
                f" vs eval {eval_punishment:.3f} |"
                f" {wall:.1f}s"
            )
            if wandb_enabled:
                wandb.log({"update_step": generation, **gen_row})
        gen_rows.append(gen_row)

    with th.no_grad():
        set_flat_params_(base.policy_model, theta.detach())
    model_file = os.path.join(model_dir, f"{config['job_id']}_manager.pt")
    print(f"Saving manager to {model_file}")
    base.save(model_file)
    base.load(model_file, device=device)

    job_id = config["job_id"]
    metrics_path = os.path.join(metrics_dir, f"{job_id}.parquet")
    df = pd.DataFrame.from_records(metrics_list)
    df = df.melt(
        id_vars=["round_number", "sampling", "update_step"],
        value_vars=VALUE_VARS,
        var_name="metric",
    )
    df = add_labels(df, {**labels, "job_id": job_id})
    df.to_parquet(metrics_path)
    print(f"Saved metrics dataframe to {metrics_path}")

    shape_path = os.path.join(metrics_dir, f"{job_id}_policy_shape.parquet")
    shape_df = pd.DataFrame.from_records(shape_rows)
    shape_df["job_id"] = job_id
    shape_df.to_parquet(shape_path)
    print(f"Saved policy shape to {shape_path}")

    gen_path = os.path.join(metrics_dir, f"{job_id}_generations.parquet")
    gen_df = pd.DataFrame.from_records(gen_rows)
    gen_df["job_id"] = job_id
    gen_df.to_parquet(gen_path)
    print(f"Saved generation diagnostics to {gen_path}")

    if wandb_enabled:
        wandb.finish()
    return model_file


def main(config):
    train_manager_es(config)
