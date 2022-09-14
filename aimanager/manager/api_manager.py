import torch as th

from aimanager.generic.graph import GraphNetwork
from aimanager.manager.manager import ArtificalManager


# def replace_none(l, dv):
#     return [dv if v is None else l for v in l]


def create_data(rounds, default_values):
    contributions = th.tensor([r['contributions'] for r in rounds], dtype=th.int64)
    valid = ~th.tensor([r['missing_inputs'] for r in rounds], dtype=th.bool)
    contributions = th.where(valid, contributions, int(default_values['contributions']))

    manager_valid = th.tensor([[p is not None for p in r['punishments']]
                              for r in rounds], dtype=th.bool)
    punishments = th.tensor(
        [[default_values['punishments'] if p is None else p for p in r['punishments']]
         for r in rounds], dtype=th.int64)
    round_number = th.tensor([[r['round']]*len(r['contributions'])
                             for r in rounds], dtype=th.int64)

    last_round = rounds[-1]
    edge_index = th.tensor([
        [g1idx, g2idx]
        for g1idx, g1 in enumerate(last_round['groups'])
        for g2idx, g2 in enumerate(last_round['groups'])
        if ((g1idx != g2idx) and (g1 == g2))], dtype=th.int64)
    batch = th.zeros(len(last_round['groups']), dtype=th.int64)

    prev_punishments = th.full_like(punishments, fill_value=default_values['punishments'])
    prev_punishments[1:] = punishments[: -1]
    prev_manager_valid = th.full_like(manager_valid, fill_value=default_values['manager_valid'])
    prev_manager_valid[1:] = manager_valid[: -1]

    data = {
        'prev_manager_valid': prev_manager_valid.T,
        'contributions': contributions.T,
        'prev_punishments': prev_punishments.T,
        'round_number': round_number.T,
        'valid': valid.T,
        'edge_index': edge_index.T,
        'batch': batch
    }
    return data


class Manager:
    def encode(self, rounds):
        data = create_data(rounds, self.model.default_values)
        # for k, v in data.items():
        #     print(k, v.shape, v.max())

        encoded = self.model.encode_pure(data, y_encode=False)
        return encoded


class HumanManager(Manager):
    def __init__(self, model_path):
        self.model = GraphNetwork.load(model_path, device=th.device('cpu'))

    def get_punishments(self, rounds):
        encoded = self.encode(rounds)
        return self.model.predict_pure(encoded, sample=True)[0][:, -1].tolist()


class RLManager(Manager):
    def __init__(self, model_path):
        self.model = ArtificalManager.load(model_path, device=th.device('cpu')).policy_model
        self.model.u_encoder.refrence = 'contributions'

    def get_punishments(self, rounds):
        encoded = self.encode(rounds)
        return self.model.predict_pure(encoded, sample=False)[0][:, -1].tolist()


MANAGER_CLASS = {
    'human': HumanManager,
    'rl': RLManager
}


class MultiManager:
    def __init__(self, managers):
        self.managers = {k: MANAGER_CLASS[m['type']](m['path']) for k, m in managers.items()}

    def get_punishments(self, rounds):
        groups = rounds[-1]['groups']
        punishments = {
            k: m.get_punishments(rounds) for k, m in self.managers.items()
        }
        return [punishments[g][i] if g in punishments else None for i, g in enumerate(groups)], punishments
