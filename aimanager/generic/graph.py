from torch.nn import Sequential as Seq, Linear as Lin, Tanh, GRU, ReLU
import numpy as np
import torch as th
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
from aimanager.generic.encoder import Encoder, IntEncoder


class EdgeModel(th.nn.Module):
    def __init__(self, x_features, edge_features, u_features, out_features):
        super().__init__()
        in_features = 2 * x_features + edge_features + u_features
        self.edge_mlp = Seq(
            Lin(in_features=in_features, out_features=out_features), Tanh()
        )

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = th.cat([src, dest, edge_attr, u[batch]], dim=-1)
        out = self.edge_mlp(out)
        return out


class NodeModel(th.nn.Module):
    def __init__(
        self, x_features, edge_features, u_features, out_features, activation=None
    ):
        super().__init__()
        in_features = x_features + edge_features + u_features
        if activation is None:
            self.node_mlp = Lin(in_features=in_features, out_features=out_features)
        else:
            self.node_mlp = Seq(
                Lin(in_features=in_features, out_features=out_features), activation
            )

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
        in_features = u_features + x_features
        self.global_mlp = Seq(
            Lin(in_features=in_features, out_features=out_features), Tanh()
        )

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
    def __init__(
        self,
        op1=None,
        op2=None,
        rnn_n=None,
        rnn_g=None,
        bias=None,
        b_encoding=None,
        *,
        y_levels=21,
        y_name="contribution",
        autoregressive=False,
        x_encoding=[],
        u_encoding=[],
        add_rnn=True,
        add_edge_model=True,
        add_global_model=True,
        hidden_size=None,
        default_values={},
        **_,
    ):
        super().__init__()
        self.x_encoder = Encoder(x_encoding, refrence=y_name)
        self.u_encoder = Encoder(u_encoding, aggregation="mean", refrence=y_name)
        self.y_encoder = IntEncoder(encoding="onehot", name=y_name, n_levels=y_levels)
        self.bias_encoder = (
            Encoder(b_encoding, refrence=y_name) if b_encoding is not None else None
        )
        self.edge_encoder = EmptyEncoder(refrence=y_name)

        x_features = self.x_encoder.size
        u_features = self.u_encoder.size
        y_features = self.y_encoder.size
        edge_features = self.edge_encoder.size
        self.x_encoding = x_encoding
        self.u_encoding = u_encoding
        self.b_encoding = b_encoding
        self.default_values = default_values
        self.y_levels = y_levels
        self.y_name = y_name
        self.autoregressive = autoregressive

        if op1 is None:
            if add_edge_model:
                edge_model = EdgeModel(
                    x_features=x_features,
                    edge_features=edge_features,
                    u_features=u_features,
                    out_features=hidden_size,
                )
                edge_features = hidden_size
            else:
                edge_model = None

            node_model = NodeModel(
                x_features=x_features,
                edge_features=edge_features,
                u_features=u_features,
                out_features=hidden_size,
                activation=Tanh(),
            )
            x_features = hidden_size

            if add_global_model:
                gobal_model = GlobalModel(
                    x_features=x_features,
                    edge_features=edge_features,
                    u_features=u_features,
                    out_features=hidden_size,
                )
                u_features = hidden_size
            else:
                gobal_model = None

            self.op1 = MetaLayer(edge_model, node_model, gobal_model)

            if add_rnn:
                self.rnn_n = GRU(
                    input_size=x_features,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                )
                self.rnn_n_h0 = None
                x_features = hidden_size
            else:
                self.rnn_n = None

            if add_rnn and add_global_model:
                self.rnn_g = GRU(
                    input_size=u_features,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                )
                self.rnn_g_h0 = None
                u_features = hidden_size
            else:
                self.rnn_g = None

            self.op2 = MetaLayer(
                None,
                NodeModel(
                    x_features=x_features,
                    edge_features=0,
                    u_features=u_features,
                    out_features=y_features,
                ),
                None,
            )
            if b_encoding is not None:
                self.bias = Seq(
                    Lin(in_features=self.bias_encoder.size, out_features=hidden_size),
                    Tanh(),
                    Lin(in_features=hidden_size, out_features=1),
                )
            else:
                self.bias = None

        else:
            self.op1 = op1
            self.op2 = op2
            self.rnn_n = rnn_n
            self.rnn_g = rnn_g
            self.bias = bias
            self.rnn_n_h0 = None
            self.rnn_g_h0 = None

    def forward(self, data, reset_rnn=True):
        x = data["x"]
        edge_index = data["edge_index"]
        if "edge_attr" in data:
            edge_attr = data["edge_attr"]
        else:
            edge_attr = th.empty(
                (edge_index.shape[1], x.shape[1], 0),
                dtype=th.float,
                device=edge_index.device,
            )
        u = data["u"]
        batch = data["batch"]
        x, _, u = self.op1(x, edge_index, edge_attr, u, batch)
        if self.rnn_n is not None:
            x, self.rnn_n_h0 = self.rnn_n(x, None if reset_rnn else self.rnn_n_h0)
        if self.rnn_g is not None:
            u, self.rnn_g_h0 = self.rnn_g(u, None if reset_rnn else self.rnn_g_h0)
        x, _, _ = self.op2(x, edge_index, edge_attr, u, batch)
        if self.bias:
            x = x + self.bias(data["b"])
        return x

    def encode(
        self,
        data,
        *,
        mask=None,
        # autoreg_mask=None,
        y_encode=True,
        edge_index=None,
        device=None,
    ):
        device = self.device if device is None else device
        if mask is not None:
            mask_ = data[mask]
        else:
            mask_ = None

        encoded = {
            "mask": mask_,
            "x": self.x_encoder(**data),
            "y_enc": self.y_encoder(**data).unsqueeze(1) if y_encode else None,
            "y": data[self.y_name] if y_encode else None,
            "u": self.u_encoder(**data, datashape="batch_agent_round"),
            **(
                {"b": self.bias_encoder(**data)}
                if self.bias_encoder is not None
                else {}
            ),
        }
        n_groups, n_player, n_rounds, _ = encoded["x"].shape
        encoded = {k: v.flatten(0, 1) for k, v in encoded.items() if v is not None}
        encoded["batch"] = th.tensor(
            [i for i in range(n_groups) for j in range(n_player)], device=device
        )
        if edge_index is None:
            edge_index = self.create_fully_connected(n_player, n_groups=n_groups)
        encoded["edge_index"] = edge_index
        encoded = {k: v.to(device) for k, v in encoded.items() if v is not None}
        return encoded

    def predict_encoded(self, data, sample=True, reset_rnn=True, temperature=1.0):
        self.eval()
        y_logit = self(data, reset_rnn)
        y_pred_proba = th.nn.functional.softmax(y_logit / temperature, dim=-1)
        y_pred = self.y_encoder.decode(y_pred_proba, sample)
        return y_pred, y_pred_proba

    def predict_independent(
        self, data, sample=True, reset_rnn=True, edge_index=None, temperature=1.0
    ):
        n_batch, n_nodes, n_rounds = data[self.y_name].shape
        if edge_index is None:
            edge_index = self.create_fully_connected(n_nodes, n_groups=n_batch)
        encoded = self.encode(
            data, y_encode=False, edge_index=edge_index, device=self.device
        )
        predict = self.predict_encoded(
            encoded, sample=sample, reset_rnn=reset_rnn, temperature=temperature
        )
        predict = tuple(t.reshape((n_batch, n_nodes, *t.shape[1:])) for t in predict)
        return predict

    def predict_autoreg(self, data, sample=True, temperature=1.0):
        self.eval()
        # print("predict autoreg")
        assert (
            self.rnn_n is None and self.rnn_g is None
        ), "Autoregressive predictions do not support RNN"

        n_batch, n_nodes, n_rounds = data["contribution"].shape
        edge_index = self.create_fully_connected(n_nodes, n_groups=n_batch)

        agent_order = np.arange(n_nodes)
        agent_order = np.random.permutation(agent_order)

        # we start with predicting all agents; we will use only the prediction
        # of one agent
        autoreg_mask = th.ones(
            (n_batch, n_nodes, n_rounds), device=self.device, dtype=th.bool
        )

        # initially set all y_pred to the default value
        y_pred = th.full_like(
            data[self.y_name], fill_value=self.default_values[self.y_name]
        )
        y_masked = data[self.y_name].clone()
        y_pred_proba = th.zeros(
            (n_batch, n_nodes, n_rounds, self.y_levels),
            device=self.device,
            dtype=th.float,
        )
        y_masked_name = self.y_name + "_masked"

        for i in agent_order:
            data[y_masked_name] = y_masked
            data["autoreg_mask"] = autoreg_mask

            # print(f"# {i}")
            # for k, v in data.items():
            #     print(k)
            #     print(v)

            encoded = self.encode(
                data,
                y_encode=False,
                edge_index=edge_index,
                device=self.device,
            )
            y_logit = self(encoded)
            y_pred_proba_ = th.nn.functional.softmax(y_logit / temperature, dim=-1)
            y_pred_ = self.y_encoder.decode(y_pred_proba_, sample)
            y_pred_ = y_pred_.reshape(n_batch, n_nodes, n_rounds)
            y_pred_proba_ = y_pred_proba_.reshape(
                n_batch, n_nodes, n_rounds, self.y_levels
            )
            y_pred[:, i] = y_pred_[:, i]
            y_pred_proba[:, i] = y_pred_proba_[:, i]
            y_masked[:, i, -1] = y_pred_[:, i, -1]
            autoreg_mask[:, i] = False

        return y_pred, y_pred_proba

    def predict(self, *args, **kwargs):
        if self.autoregressive:
            return self.predict_autoreg(*args, **kwargs)
        else:
            return self.predict_independent(*args, **kwargs)

    def save(self, filename):
        to_save = [
            "op1",
            "op2",
            "rnn_n",
            "rnn_g",
            "bias",
            "y_levels",
            "y_name",
            "autoregressive",
            "x_encoding",
            "u_encoding",
            "b_encoding",
            "default_values",
        ]
        th.save({k: getattr(self, k) for k in to_save}, filename)

    @classmethod
    def load(cls, filename, device=None):
        to_load = th.load(filename, map_location=device)
        ah = cls(**to_load, device=device)
        ah.device = device
        return ah

    def to(self, device):
        self.device = device
        return super().to(device)

    def create_fully_connected(self, n_nodes, n_groups=1):
        return th.tensor(
            [
                [i + k * n_nodes, j + k * n_nodes]
                for k in range(n_groups)
                for i in range(n_nodes)
                for j in range(n_nodes)
                if i != j
            ],
            device=self.device,
        ).T
