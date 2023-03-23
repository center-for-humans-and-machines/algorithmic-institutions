from aimanager.generic.graph_encode import create_fully_connected

import torch as th
from torch_scatter import scatter_sum


class ArtificialHumanEnv:
    """
    Environment that runs the virtual humans and calculuates the value of the common good.

    Indices:
        t: agent types [0..1]
    """

    state_dimensions = {
        "punishment": ["agent"],
        "contribution": ["agent"],
        "payoffs": ["agent"],
        "contribution_valid": ["agent"],
        "common_good": ["agent"],
        "round_number": ["agent"],
        "player_id": ["agent"],
    }

    def __init__(
        self,
        *,
        artifical_humans,
        artifical_humans_valid=None,
        batch_size,
        n_agents,
        n_contributions,
        n_punishments,
        n_rounds,
        device,
        default_values=None,
    ):
        """
        Args:
            asdasd
        """
        self.batch_size = batch_size
        self.default_values = (
            artifical_humans.default_values
            if default_values is None
            else default_values
        )
        self.n_rounds = n_rounds
        self.device = device
        self.n_contributions = n_contributions
        self.n_punishments = n_punishments
        self.artifical_humans = artifical_humans
        self.artifical_humans_valid = artifical_humans_valid
        self.n_agents = n_agents
        self.edge_index = create_fully_connected(n_agents)
        self.batch_edge_index = th.tensor(
            [
                [a + (i * self.n_agents), b + (i * self.n_agents)]
                for i in range(self.batch_size)
                for a in range(self.n_agents)
                for b in range(self.n_agents)
                if a != b
            ],
            device=self.device,
            dtype=th.int64,
        ).T

        self.batch = th.tensor(
            [i for i in range(self.batch_size) for a in range(self.n_agents)],
            device=self.device,
            dtype=th.int64,
        )
        self.groups = [
            [(i * self.n_agents + a) for a in range(self.n_agents)]
            for i in range(self.batch_size)
        ]

        self.reset_state()

    def reset_state(self):
        size = (self.batch_size, self.n_agents, 1)
        state = {
            "punishment": th.zeros(size, dtype=th.int64, device=self.device),
            "contribution": th.zeros(size, dtype=th.int64, device=self.device),
            "round_number": th.zeros(size, dtype=th.int64, device=self.device),
            "is_first": th.ones(size, dtype=th.bool, device=self.device),
            "contribution_valid": th.zeros(size, dtype=th.bool, device=self.device),
            "punishment_valid": th.zeros(size, dtype=th.bool, device=self.device),
            "common_good": th.zeros(size, dtype=th.float, device=self.device),
            "contributor_payoff": th.zeros(size, dtype=th.float, device=self.device),
            "manager_payoff": th.zeros(size, dtype=th.float, device=self.device),
            "reward": th.zeros(size, dtype=th.float, device=self.device),
            "group": th.tensor(
                [[i for a in g] for i, g in enumerate(self.groups)],
                dtype=th.int64,
                device=self.device,
            ),
            "agent": th.tensor(
                [[a for a in g] for i, g in enumerate(self.groups)],
                dtype=th.int64,
                device=self.device,
            ),
        }

        prev_state = {
            f"prev_{k}": th.full_like(state[k], fill_value=self.default_values[k])
            for k, t in state.items()
            if k in self.default_values
        }
        self.state = {**prev_state, **state}

    def __getattr__(self, name):
        if "state" in self.__dict__:
            state = self.__dict__["state"]
            return state[name]

    def __setattr__(self, name, value):
        if "state" in self.__dict__:
            if name in self.__dict__["state"]:
                assert (
                    value.shape == self.state[name].shape
                ), f"Shape of {name} does not match. [{value.shape} != {self.state[name].shape}]"
                self.state[name] = value
            else:
                object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    def update_common_good(self):
        masked_contribution = th.where(self.contribution_valid, self.contribution, 0)
        masked_punishment = th.where(self.contribution_valid, self.punishment, 0)
        sum_contribution = masked_contribution.sum(dim=1, keepdim=True)
        sum_punishment = masked_punishment.sum(dim=1, keepdim=True)
        sum_contribution_valid = self.contribution_valid.sum(dim=1, keepdim=True)
        self.common_good = (
            (sum_contribution * 1.6 - sum_punishment) / sum_contribution_valid
        ).expand(-1, self.n_agents, -1)

    def update_payoff(self):
        contributor_payoff = 20 - self.contribution - self.punishment + self.common_good
        self.contributor_payoff = th.where(
            self.contribution_valid, contributor_payoff, 0
        )
        self.manager_payoff = self.common_good / 4

    def update_reward(self):
        masked_contribution = th.where(self.contribution_valid, self.contribution, 0)
        masked_prev_punishment = th.where(
            self.prev_contribution_valid, self.prev_punishment, 0
        )

        if self.done:
            self.reward = -masked_prev_punishment.to(th.float) / 32
        else:
            self.reward = (masked_contribution * 1.6 - masked_prev_punishment) / 32

    def update_contribution(self):
        contribution = self.artifical_humans.predict(
            self.state,
            reset_rnn=self.round_number[0, 0, 0] == 0,
            edge_index=self.batch_edge_index,
        )[0]

        # artificial humans valid
        if self.artifical_humans_valid is not None:
            contribution_valid = self.artifical_humans_valid.predict(
                self.state,
                reset_rnn=self.round_number[0, 0, 0] == 0,
                edge_index=self.batch_edge_index,
            )[0]
            contribution_valid = contribution_valid.to(th.bool)
            contribution[~contribution_valid] = self.artifical_humans.default_values[
                "contribution"
            ]
        else:
            contribution_valid = th.ones_like(self.contribution_valid)

        self.contribution = contribution
        self.contribution_valid = contribution_valid

    def reset(self):
        self.round_number = th.zeros_like(self.round_number)
        self.done = False
        self.reset_state()
        self.update_contribution()
        return self.state

    def punish(self, punishment):
        assert punishment.max() < self.n_punishments
        assert punishment.dtype == th.int64
        self.punishment = punishment
        self.punishment_valid = th.ones_like(self.punishment_valid)
        self.update_common_good()
        # self.update_payoff()
        return self.state

    def step(self):
        self.round_number += 1
        self.is_first = th.zeros_like(self.is_first)
        if self.done:
            raise ValueError("Environment is done already.")
        for k in self.state:
            if k[:4] == "prev":
                self.state[k] = self.state[k[5:]]
        if self.round_number[0, 0] == (self.n_rounds):
            self.done = True
        else:
            self.update_contribution()
        self.update_reward()
        return self.state, self.reward, self.done
