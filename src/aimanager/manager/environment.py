import torch as th


def create_fully_connected(n_nodes):
    return th.tensor(
        [[i, j] for i in range(n_nodes) for j in range(n_nodes) if i != j]
    ).T


class ArtificialHumanEnv:
    """
    Environment that runs the virtual humans and
    calculuates the value of the common good.

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
        artifical_humans_switch=None,
        switch_every=None,
        batch_size,
        n_agents,
        n_contributions,
        n_punishments,
        n_rounds,
        device,
        n_groups=1,
        agent_groups=None,
        default_values=None,
    ):
        """
        Args:
            artifical_humans: The virtual humans that will be used to generate
                the contribution.
            artifical_humans_valid: The virtual humans that will be used to
                generate the action validity.
            artifical_humans_switch: The virtual humans that will be used to
                predict group switching.
            switch_every: Number of rounds between switching decisions.
                Switching occurs at rounds where (round_number+1) %
                switch_every == 0.
            batch_size: The number of batches.
            n_agents: The number of agents.
            n_contributions: The number of contributions.
            n_punishments: The number of punishments.
            n_rounds: The number of rounds.
            device: The device to use.
            default_values: The default values for the state.
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
        self.n_groups = n_groups
        self.artifical_humans = artifical_humans
        self.artifical_humans_valid = artifical_humans_valid
        self.artifical_humans_switch = artifical_humans_switch
        self.switch_every = switch_every
        self.n_agents = n_agents
        self.batch = th.tensor(
            [i for i in range(self.batch_size) for a in range(self.n_agents)],
            device=self.device,
            dtype=th.int64,
        )

        if agent_groups is not None:
            agent_groups = th.tensor(
                agent_groups, device=device, dtype=th.int64
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            agent_groups = th.arange(
                n_agents, device=device
            ).div(
                n_agents // n_groups, rounding_mode="floor"
            ).clamp(
                max=n_groups - 1
            ).unsqueeze(0).expand(batch_size, -1)
        self.update_groups(agent_groups)
        self.reset_state()

        self.reset()

    def update_groups(self, agent_groups):
        """
        Updates the groups of agents.

        Args:
            agent_groups: (batch_size, n_agents) tensor
            containing the group of each agent.
        """
        self.agent_groups = agent_groups.unsqueeze(-1)
        self.batch_edge_index = th.tensor(
            [
                [a + (i * self.n_agents), b + (i * self.n_agents)]
                for i in range(self.batch_size)
                for a in range(self.n_agents)
                for b in range(self.n_agents)
                if (a != b) and (agent_groups[i, a] == agent_groups[i, b])
            ],
            device=self.device,
            dtype=th.int64,
        ).T
        self.agent_group_mask = th.nn.functional.one_hot(
            agent_groups, num_classes=self.n_groups
        ).unsqueeze(-1)

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
            # "manager_payoff": th.zeros(size, dtype=th.float, device=self.device),
            "reward": th.zeros(
                (self.batch_size, self.n_groups, 1), dtype=th.float, device=self.device
            ),
            "agent_group": self.agent_groups.clone()
            if hasattr(self, "agent_groups")
            else th.zeros(size, dtype=th.int64, device=self.device),
            "group_payoff": th.zeros(
                (self.batch_size, self.n_groups, 1), dtype=th.float, device=self.device
            ),
            "does_switch": th.zeros(size, dtype=th.bool, device=self.device),
            "switch_mask": th.zeros(size, dtype=th.bool, device=self.device),
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
                assert value.shape == self.state[name].shape, (
                    f"Shape of {name} does not match. "
                    f"[{value.shape} != {self.state[name].shape}]"
                )
                self.state[name] = value
            else:
                object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    def compute_common_good_per_group(
        self, contribution, punishment, contribution_valid
    ):
        # Method is used with punishment and prev_punishment
        # in update_common_good and update_reward
        # Set the contribution and punishment to 0 if they are not valid
        contribution = th.where(
            contribution_valid, contribution, th.zeros_like(contribution)
        )
        punishment = th.where(contribution_valid, punishment, th.zeros_like(punishment))

        # Add a dimension for the groups
        contribution = contribution.unsqueeze(-2) * self.agent_group_mask
        punishment = punishment.unsqueeze(-2) * self.agent_group_mask
        contribution_valid = contribution_valid.unsqueeze(-2) * self.agent_group_mask

        # Sum over the agents for each group
        sum_contribution = contribution.sum(dim=1)
        sum_punishment = punishment.sum(dim=1)
        sum_contribution_valid = contribution_valid.sum(dim=1)

        # Calculate the common good per group
        common_good_per_group = (
            sum_contribution * 1.6 - sum_punishment
        ) / sum_contribution_valid
        # Set common good to 0 if no valid contributions
        common_good_per_group = th.where(
            sum_contribution_valid > 0,
            common_good_per_group,
            th.zeros_like(common_good_per_group),
        )
        return common_good_per_group

    def compute_average_payoff_per_group(
        self, contribution, punishment, contribution_valid, common_good
    ):
        # Used both with punishment and prev_punishment
        # in update_payoff and update_reward
        # Compute the payoff for the contributors
        contributor_payoff = 20 - contribution - punishment + common_good

        # Set the payoff to 0 if the contribution is not valid
        contributor_payoff = th.where(
            contribution_valid, contributor_payoff, th.zeros_like(contributor_payoff)
        )

        # Compute the average payoff for each group
        average_payoff_per_group = (
            contributor_payoff.unsqueeze(-2) * self.agent_group_mask
        )
        contribution_valid = contribution_valid.unsqueeze(-2) * self.agent_group_mask

        average_payoff_per_group = average_payoff_per_group.sum(
            dim=1
        ) / contribution_valid.sum(dim=1)
        average_payoff_per_group = th.where(
            contribution_valid.sum(dim=1) > 0,
            average_payoff_per_group,
            th.zeros_like(average_payoff_per_group),
        )
        return contributor_payoff, average_payoff_per_group

    def compute_average_per_group(self, metric, valid=None):
        if valid is None:
            valid = th.ones_like(metric, dtype=th.bool)

        sum_per_group = metric.unsqueeze(-2) * self.agent_group_mask
        valid_per_group = valid.unsqueeze(-2) * self.agent_group_mask

        average_per_group = sum_per_group.sum(dim=1) / valid_per_group.sum(dim=1)

        average_per_group = th.where(
            valid_per_group.sum(dim=1) > 0,
            average_per_group,
            th.zeros_like(average_per_group),
        )
        return average_per_group

    def update_common_good(self):
        common_good_per_group = self.compute_common_good_per_group(
            self.contribution, self.punishment, self.contribution_valid
        )

        # Broadcast the common good of each group to the agents
        common_good_per_agent = common_good_per_group.gather(1, self.agent_groups)
        self.common_good = common_good_per_agent

    def update_payoff(self):

        self.contributor_payoff, self.group_payoff = (
            self.compute_average_payoff_per_group(
                self.contribution,
                self.punishment,
                self.contribution_valid,
                self.common_good,
            )
        )

    def update_reward(self):
        self.reward = self.group_payoff

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
        assert self.state is not None
        assert punishment.max() < self.n_punishments
        assert punishment.dtype == th.int64
        self.punishment = punishment
        self.punishment_valid = th.ones_like(self.punishment_valid)
        self.update_common_good()
        self.update_payoff()
        return self.state

    def update_groups_from_switch_predictor(self):
        """Use the switch predictor to update group assignments."""
        does_switch = self.artifical_humans_switch.predict(
            self.state,
            reset_rnn=self.round_number[0, 0, 0] == 0,
            edge_index=self.batch_edge_index,
        )[0]
        does_switch = does_switch.squeeze(-1).to(th.bool)

        # Flip group: 0->1, 1->0
        current_groups = self.agent_groups.squeeze(-1)
        new_groups = th.where(
            does_switch, 1 - current_groups, current_groups
        )
        self.update_groups(new_groups)
        self.state["agent_group"] = self.agent_groups.clone()

    def step(self):
        assert self.state is not None
        self.round_number += 1
        self.is_first = th.zeros_like(self.is_first)
        if self.done:
            raise ValueError("Environment is done already.")
        for k in self.state:
            if k[:4] == "prev":
                self.state[k] = self.state[k[5:]]

        # Group switching at decision rounds
        if (
            self.artifical_humans_switch is not None
            and self.switch_every is not None
            and self.round_number[0, 0, 0] % self.switch_every == 0
        ):
            self.update_groups_from_switch_predictor()

        if self.round_number[0, 0] == (self.n_rounds):
            self.done = True
        else:
            self.update_contribution()
        self.update_reward()
        return self.state, self.reward, self.done
