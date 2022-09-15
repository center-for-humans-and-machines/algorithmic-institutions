
import torch as th
from torch_geometric.data import Data


def create_fully_connected(n_nodes):
    return th.tensor([[i, j]
                      for i in range(n_nodes)
                      for j in range(n_nodes)
                      if i != j
                      ]).T


def encode(
        model, data, *, mask=True, index=False, y_encode=True, device, n_player=4, syn_index=[]):
    encoded = model.encode(data, mask=mask, y_encode=y_encode)

    encoded['info'] = th.stack([data[c] for c in syn_index], dim=-1) if index else None

    n_episodes, n_agents, n_rounds, _ = encoded['x'].shape

    edge_attr = th.zeros(n_player*n_player, n_rounds, 0)
    edge_index = create_fully_connected(n_player)

    dataset = [
        Data(
            **{k: v[i] for k, v in encoded.items() if v is not None},
            edge_attr=edge_attr, edge_index=edge_index, idx=i, group_idx=i, num_nodes=n_player
        ).to(device)
        for i in range(n_episodes)
    ]
    return dataset
