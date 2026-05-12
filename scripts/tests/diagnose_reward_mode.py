"""Side-by-side dump of env reward fields under reward_mode=avg vs sum.

Builds the same 6-agent / 2-group / asymmetric env used in the unit tests,
runs `punish` then `step`, and prints expected vs actual values so a reviewer
can eyeball that:
  * `group_payoff` (avg) and `group_payoff_sum` are logged in BOTH modes,
  * `reward` selects `group_payoff` under `avg` and `group_payoff_sum`
    under `sum`.

Run on Raven via scripts/remote_test.sh (PyG/torch-scatter is Linux-only).
"""

import torch as th

from aimanager.manager.environment import ArtificialHumanEnv


class MockAH:
    def __init__(self, default_values):
        self.default_values = default_values

    def predict(self, state, reset_rnn, edge_index):
        return (th.ones(state["contribution"].shape, dtype=th.long),)


class MockAHValid:
    def predict(self, state, reset_rnn, edge_index):
        return (th.ones(state["contribution"].shape, dtype=th.bool),)


def build_env(reward_mode):
    defaults = {
        "punishment": 1,
        "contribution": 1,
        "payoffs": 1,
        "contribution_valid": True,
        "common_good": 1,
        "round_number": 1,
        "player_id": 1,
    }
    return ArtificialHumanEnv(
        artifical_humans=MockAH(defaults),
        artifical_humans_valid=MockAHValid(),
        batch_size=1,
        n_agents=6,
        n_contributions=3,
        n_punishments=3,
        n_rounds=2,
        n_groups=2,
        device="cpu",
        agent_groups=[0, 0, 0, 0, 1, 1],
        reward_mode=reward_mode,
        default_values={
            "punishment": 0,
            "contribution": 0,
            "round_number": 0,
            "is_first": False,
            "contribution_valid": False,
            "punishment_valid": False,
            "common_good": 0,
            "contributor_payoff": 0,
            "manager_payoff": 0,
            "reward": 0,
        },
    )


def main():
    punishment = th.tensor([[[1], [0], [2], [1], [0], [2]]])

    # Hand-computed expected values:
    # group 0 (4 agents, punish=[1,0,2,1]): cg = (4*1.6-4)/4 = 0.6
    #   payoffs [18.6, 19.6, 17.6, 18.6] -> sum 74.4, avg 18.6
    # group 1 (2 agents, punish=[0,2]):    cg = (2*1.6-2)/2 = 0.6
    #   payoffs [19.6, 17.6]              -> sum 37.2, avg 18.6
    expected_avg = th.tensor([[[18.6], [18.6]]])
    expected_sum = th.tensor([[[74.4], [37.2]]])

    print("=== reward_mode comparison (group sizes: 4, 2) ===")
    print(f"punishment        : {punishment.flatten().tolist()}")
    print(f"expected avg/grp  : {expected_avg.flatten().tolist()}")
    print(f"expected sum/grp  : {expected_sum.flatten().tolist()}")

    for mode in ("avg", "sum"):
        env = build_env(mode)
        env.punish(punishment)

        gp = env.group_payoff.flatten().tolist()
        gps = env.group_payoff_sum.flatten().tolist()
        _, reward, _ = env.step()
        r = reward.flatten().tolist()

        print(f"\n--- reward_mode={mode!r} ---")
        print(f"  state['group_payoff']     (avg) = {gp}")
        print(f"  state['group_payoff_sum']       = {gps}")
        print(f"  reward returned by step()       = {r}")

        ok_avg = th.allclose(env.group_payoff, expected_avg)
        ok_sum = th.allclose(env.group_payoff_sum, expected_sum)
        ok_rwd = th.allclose(reward, expected_sum if mode == "sum" else expected_avg)
        print(f"  avg field correct : {ok_avg}")
        print(f"  sum field correct : {ok_sum}")
        print(f"  reward selects {mode} : {ok_rwd}")


if __name__ == "__main__":
    main()
