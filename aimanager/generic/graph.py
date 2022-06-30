from torch.nn import Sequential as Seq, Linear as Lin, Tanh, GRU, ReLU
import torch as th
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
from aimanager.generic.encoder import Encoder, IntEncoder
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data


class EdgeModel(th.nn.Module):
    def __init__(self, x_features, edge_features, u_features, out_features):
        super().__init__()
        in_features = 2*x_features+edge_features+u_features
        self.edge_mlp = Seq(Lin(in_features=in_features, out_features=out_features), Tanh())

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = th.cat([src, dest, edge_attr, u[batch]], dim=-1)
        out = self.edge_mlp(out)
        return out


class NodeModel(th.nn.Module):
    def __init__(self, x_features, edge_features, u_features, out_features, activation=None):
        super().__init__()
        in_features = x_features+edge_features+u_features
        if activation is None:
            self.node_mlp = Lin(in_features=in_features, out_features=out_features)
        else:
            self.node_mlp = Seq(Lin(in_features=in_features, out_features=out_features), activation)


    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        row, col = edge_index
        out = scatter_mean(edge_attr, col, dim=0, dim_size=x.size(0))
        out = th.cat([x, out, u[batch]], dim=-1)
        out = self.node_mlp(out)
        return out

class GlobalModel(th.nn.Module):
    def __init__(self, x_features, edge_features, u_features, out_features):
        super().__init__()
        in_features = u_features+x_features
        self.global_mlp = Seq(Lin(in_features=in_features, out_features=out_features), Tanh())

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = th.cat([u, scatter_mean(x, batch, dim=0)], dim=-1)
        return self.global_mlp(out)


class EmptyEncoder(th.nn.Module):
    def __init__(self, refrence):
        super(EmptyEncoder, self).__init__()
        self.size = 0
        self.refrence = refrence

    def forward(self, *, n_edges, **state):
        n_episodes, n_agents, n_rounds = state[self.refrence].shape
        return th.empty((n_episodes, n_edges, n_rounds, 0), dtype=th.float)


class GraphNetwork(th.nn.Module):
    def __init__(self, n_contributions, n_punishments, x_encoding=[], u_encoding=[], add_rnn=True, add_edge_model=True, 
            add_global_model=True, hidden_size=None, op1=None, op2=None, rnn_n=None, rnn_g=None, default_values={}):
        super().__init__()
        self.x_encoder = Encoder(x_encoding)
        self.u_encoder = Encoder(u_encoding, aggregation='mean')
        self.y_encoder = IntEncoder(encoding='onehot', name='contributions', n_levels=n_contributions)
        self.edge_encoder = EmptyEncoder(refrence='contributions')

        x_features = self.x_encoder.size
        u_features = self.u_encoder.size
        y_features = self.y_encoder.size
        edge_features = self.edge_encoder.size
        self.n_contributions = n_contributions
        self.n_punishments = n_punishments
        self.x_encoding = x_encoding
        self.u_encoding = u_encoding
        self.default_values = default_values

        if op1 is None:
            if add_edge_model:
                edge_model = EdgeModel(
                    x_features=x_features, edge_features=edge_features, 
                    u_features=u_features, out_features=hidden_size)
                edge_features = hidden_size
            else:
                edge_model = None

            node_model = NodeModel(
                x_features=x_features, edge_features=edge_features, 
                u_features=u_features, out_features=hidden_size, activation=ReLU())
            x_features = hidden_size

            if add_global_model:
                gobal_model = GlobalModel(
                    x_features=x_features, edge_features=edge_features, 
                    u_features=u_features, out_features=hidden_size)
                u_features = hidden_size
            else:
                gobal_model = None

            self.op1 = MetaLayer(edge_model, node_model, gobal_model)

            if add_rnn:
                self.rnn_n = GRU(input_size=x_features, hidden_size=hidden_size, num_layers=1, batch_first=True)
                self.rnn_n_h0 = None
                x_features = hidden_size
            else:
                self.rnn_n = None

            if add_rnn and add_global_model:
                self.rnn_g = GRU(input_size=u_features, hidden_size=hidden_size, num_layers=1, batch_first=True)
                self.rnn_g_h0 = None
                u_features = hidden_size
            else:
                self.rnn_g = None


            self.op2 = MetaLayer(
                None,
                NodeModel(
                    x_features=x_features, edge_features=0, 
                    u_features=u_features, out_features=y_features), 
                None
            )
        else:
            self.op1 = op1
            self.op2 = op2
            self.rnn_n = rnn_n
            self.rnn_g = rnn_g
            self.rnn_n_h0 = None
            self.rnn_g_h0 = None
            
    

    def encode(self, data, *, edge_index, mask=True, y_encode=True, info_columns=None):
        encoded = {
            'mask': data['valid'] if mask else None,
            'x': self.x_encoder(**data),
            'y_enc': self.y_encoder(**data) if y_encode else None,
            'y': data['contributions'] if y_encode else None,
            'u': self.u_encoder(**data),
            'edge_attr': self.edge_encoder(**data, n_edges=edge_index.shape[1]),
            'info': th.stack([data[c] for c in info_columns], dim=-1) if info_columns else None
        }
        n_episodes, n_agents, n_rounds, _ = encoded['x'].shape

        dataset = [
            Data(
                **{k: v[i] for k, v in encoded.items() if v is not None}, edge_index=edge_index, idx=i, group_idx=i, num_nodes=n_agents
                ).to(self.device)
            for i in range(n_episodes)
        ]
        return dataset

    def forward(self, data, reset_rnn=True):
        x = data['x']
        edge_index = data['edge_index']
        edge_attr = data['edge_attr']
        u = data['u']
        batch = data['batch']
        x, _, u = self.op1(x, edge_index, edge_attr, u, batch)
        if self.rnn_n is not None:
            x, self.rnn_n_h0 = self.rnn_n(x, None if reset_rnn else self.rnn_n_h0)
        if self.rnn_g is not None:
            u, self.rnn_g_h0 = self.rnn_g(u, None if reset_rnn else self.rnn_g_h0)
        x, _, _ = self.op2(x, edge_index, edge_attr, u, batch)
        return x

    def predict(self, data):
        self.eval()
        y_pred_logit = th.cat([self(d)
            for d in iter(DataLoader(data, shuffle=False, batch_size=10))
        ])
        y_pred_proba = th.nn.functional.softmax(y_pred_logit, dim=-1)
        y_pred = self.y_encoder.decode(y_pred_proba)
        return y_pred, y_pred_proba


    def predict_one(self, data, reset_rnn=True, sample=True):
        self.eval()
        batch = Batch.from_data_list([data])
        y_pred_logit = self(batch, reset_rnn)
        y_pred_proba = th.nn.functional.softmax(y_pred_logit, dim=-1)
        y_pred = self.y_encoder.decode(y_pred_proba, sample)
        return y_pred, y_pred_proba


    def save(self, filename):
        to_save = {
            'op1': self.op1,
            'op2': self.op2,
            'n_contributions': self.n_contributions,
            'n_punishments': self.n_punishments,
            'x_encoding': self.x_encoding, 
            'u_encoding': self.u_encoding,
            'default_values': self.default_values
        }
        th.save(to_save, filename)

    @classmethod
    def load(cls, filename):
        to_load = th.load(filename)
        ah = cls(**to_load)
        return ah

    def to(self, device):
        self.device = device
        return super().to(device)