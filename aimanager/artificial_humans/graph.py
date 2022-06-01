from torch.nn import Sequential as Seq, Linear as Lin, Tanh, GRU, ReLU
import torch as th
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
from aimanager.generic.encoder import Encoder, IntEncoder
from torch_geometric.loader import DataLoader


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

class GraphNetwork(th.nn.Module):
    def __init__(self, n_contributions, n_punishments, x_encoding=[], u_encoding=[], add_rnn=True, add_edge_model=True, 
            add_global_model=True, hidden_size=None, op1=None, op2=None, rnn_n=None, rnn_g=None):
        super().__init__()
        self.x_encoder = Encoder(x_encoding)
        self.u_encoder = Encoder(u_encoding, aggregation='mean')
        self.y_encoder = IntEncoder(encoding='onehot', name='contributions', n_levels=n_contributions)
        x_features = self.x_encoder.size
        u_features = self.u_encoder.size
        y_features = self.y_encoder.size
        self.n_contributions = n_contributions
        self.n_punishments = n_punishments
        self.x_encoding = x_encoding
        self.u_encoding = u_encoding

        edge_features = 0
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
                x_features = hidden_size
            else:
                self.rnn_n = None

            if add_rnn and add_global_model:
                self.rnn_g = GRU(input_size=u_features, hidden_size=hidden_size, num_layers=1, batch_first=True)
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
    
    def forward(self, data):
        x = data['x']
        edge_index = data['edge_index']
        edge_attr = data['edge_attr']
        u = data['u']
        batch = data['batch']
        x, _, u = self.op1(x, edge_index, edge_attr, u, batch)
        if self.rnn_n is not None:
            x, x_h_n = self.rnn_n(x)
        if self.rnn_g is not None:
            u, u_h_n = self.rnn_g(u)
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

    def save(self, filename):
        to_save = {
            'op1': self.op1,
            'op2': self.op2,
            'n_contributions': self.n_contributions,
            'n_punishments': self.n_punishments,
            'x_encoding': self.x_encoding, 
            'u_encoding': self.u_encoding
        }
        th.save(to_save, filename)

    @classmethod
    def load(cls, filename):
        to_load = th.load(filename)
        ah = cls(**to_load)
        return ah
