from aimanager.manager.environment import ArtificialHumanEnv
from operator import itemgetter


class ArtificialHumanGroup:
    def __init__(self, human_groups: list[ArtificialHumanEnv]):
        self.human_groups = human_groups

    @classmethod
    def build_group(
        cls,
        num_groups,
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
        assert batch_size == 1, "only batch_size=1 is supported for now."
        return cls(
            [
                ArtificialHumanEnv(
                    artifical_humans=artifical_humans,
                    artifical_humans_valid=artifical_humans_valid,
                    batch_size=batch_size,
                    n_agents=n_agents,
                    n_contributions=n_contributions,
                    n_punishments=n_punishments,
                    n_rounds=n_rounds,
                    device=device,
                    default_values=default_values,
                )
                for _ in range(num_groups)
            ]
        )

    def step(self):
        for group in self.human_groups:
            group.step()

    def _get_better_avg_movement_map(self):
        avg_payoffs = [group.average_contributor_payoff for group in self.human_groups]
        max_idx, max_avg = max(enumerate(avg_payoffs), key=itemgetter(1))
        move_map: dict[tuple, int] = {}
        for gid, group in enumerate(self.human_groups):
            for aid in range(group.n_agents):
                if group.contributor_payoff[0, aid] < max_avg:
                    move_map[(gid, aid)] = max_idx
                else:
                    move_map[(gid, aid)] = gid

    def do_group_selection(self, strategy):
        if strategy == "move_to_better_avg":
            pass
