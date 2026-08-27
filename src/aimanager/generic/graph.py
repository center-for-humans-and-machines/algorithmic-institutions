from torch.nn import Sequential as Seq, Linear as Lin, Tanh, GRU
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


class SameGroupEdgeEncoder(th.nn.Module):
    """Derived per-edge feature: 1.0 if the two endpoints share a sub-group.

    Relational, so it cannot use the per-node encoders in ``encoder.py`` (those
    map one named node tensor to a feature axis). It reads ``agent_group`` at
    both endpoints via ``edge_index``. ``agent_group`` is time-varying (agents
    switch groups), so the bit is computed per round.
    """

    def __init__(self, name="same_group", etype="bool", **_):
        super().__init__()
        assert etype == "bool", f"same_group must be etype 'bool', got {etype}"
        self.name = name
        self.size = 1

    def forward(self, *, edge_index, **state):
        # agent_group: (N, n_rounds) int64, flattened with per-batch node offsets
        # so edge_index entries gather directly into dim 0 (no batch handling).
        ag = state["agent_group"]
        row, col = edge_index
        same = ag[row] == ag[col]  # (E, n_rounds) bool
        return same.float().unsqueeze(-1)  # (E, n_rounds, 1)


class ARPunishmentEdgeEncoder(th.nn.Module):
    """Autoregressive within-round conditioning on groupmate punishments.

    Two channels per edge and round: a gate that is on only where source and
    destination share the sub-group at that round *and* the source agent's
    punishment is already decided (``autoreg_mask`` false), and the gated,
    normalised punishment of the source agent. The channel therefore carries
    only the manager's own already-decided punishments of the same round and
    the same agent group -- never the other group's punishments (the graph is
    fully connected across sub-groups) and never any current-round
    contribution.

    Relational like ``SameGroupEdgeEncoder``, so it reads its state at both
    endpoints via ``edge_index`` instead of using the per-node encoders.
    """

    masked_key = "punishment_masked"
    mask_key = "autoreg_mask"

    def __init__(self, name="ar_punishment", n_levels=31, **_):
        super().__init__()
        assert n_levels > 1, f"ar_punishment needs n_levels > 1, got {n_levels}"
        self.name = name
        self.n_levels = n_levels
        self.size = 2

    def forward(self, *, edge_index, **state):
        # agent_group, punishment_masked, autoreg_mask: (N, n_rounds), each
        # flattened with per-batch node offsets so edge_index gathers endpoints
        # directly (no batch handling).
        for key in ("agent_group", self.masked_key, self.mask_key):
            assert key in state, (
                f"ar_punishment requires '{key}' in the edge state; the data "
                "passed to encode() must carry agent_group, "
                f"{self.masked_key} and {self.mask_key} "
                "(see apply_mask_pattern / predict_autoreg)"
            )
        ag = state["agent_group"]
        y_masked = state[self.masked_key]
        undecided = state[self.mask_key].bool()
        row, col = edge_index
        gate = (ag[row] == ag[col]) & ~undecided[row]  # (E, n_rounds) bool
        gate = gate.float().unsqueeze(-1)  # (E, n_rounds, 1)
        value = y_masked[row].float().unsqueeze(-1) / (self.n_levels - 1)
        return th.cat([gate, gate * value], dim=-1)  # (E, n_rounds, 2)


EDGE_ENCODERS = {
    "same_group": SameGroupEdgeEncoder,
    "ar_punishment": ARPunishmentEdgeEncoder,
}


class EdgeEncoder(th.nn.Module):
    """Builds per-edge features from an ``edge_encoding`` config list.

    Each entry is dispatched by ``name`` to a derived edge encoder (see
    ``EDGE_ENCODERS``). An empty list reports ``size == 0`` and emits an empty
    ``(E, n_rounds, 0)`` tensor, so configs/models without ``edge_encoding``
    behave exactly as before (the edge MLP receives no edge features).
    """

    def __init__(self, edge_encoding, refrence):
        super().__init__()
        self.refrence = refrence
        self.encoder = th.nn.ModuleList(
            [EDGE_ENCODERS[e["name"]](**e) for e in edge_encoding]
        )
        self.size = sum(e.size for e in self.encoder)

    def forward(self, *, edge_index, n_rounds, **state):
        if len(self.encoder) == 0:
            return th.empty(
                (edge_index.shape[1], n_rounds, 0),
                dtype=th.float,
                device=edge_index.device,
            )
        encoding = [e(edge_index=edge_index, **state) for e in self.encoder]
        return th.cat(encoding, dim=-1)


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
        copula_rho=None,
        x_encoding=[],
        u_encoding=[],
        edge_encoding=[],
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
        self.edge_encoder = EdgeEncoder(edge_encoding or [], refrence=y_name)

        x_features = self.x_encoder.size
        u_features = self.u_encoder.size
        y_features = self.y_encoder.size
        edge_features = self.edge_encoder.size
        self.x_encoding = x_encoding
        self.u_encoding = u_encoding
        self.edge_encoding = edge_encoding
        self.b_encoding = b_encoding
        self.default_values = default_values
        self.y_levels = y_levels
        self.y_name = y_name
        self.autoregressive = autoregressive
        # Severity-copula weight for autoregressive sampling. None (absent from
        # a legacy checkpoint) and 0.0 both mean "off" and must leave the
        # sampling path -- values and RNG stream -- untouched.
        assert copula_rho is None or (
            isinstance(copula_rho, float) and 0.0 <= copula_rho < 1.0
        ), f"copula_rho must be None or a float in [0, 1), got {copula_rho!r}"
        self.copula_rho = copula_rho

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
        # op2 is a readout with no edge model (edge_features=0), but NodeModel
        # always aggregates edge_attr -- feed it an empty one so a non-empty
        # edge feature consumed by op1 does not leak into the readout's widths.
        op2_edge_attr = th.empty(
            (edge_index.shape[1], x.shape[1], 0),
            dtype=edge_attr.dtype,
            device=edge_attr.device,
        )
        x, _, _ = self.op2(x, edge_index, op2_edge_attr, u, batch)
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
        n_batch, n_player, n_rounds, _ = encoded["x"].shape
        encoded = {k: v.flatten(0, 1) for k, v in encoded.items() if v is not None}
        encoded["batch"] = th.tensor(
            [i for i in range(n_batch) for j in range(n_player)], device=device
        )
        if edge_index is None:
            edge_index = self.create_fully_connected(n_player, n_batch=n_batch)
        encoded["edge_index"] = edge_index
        encoded = {k: v.to(device) for k, v in encoded.items() if v is not None}
        # Per-edge features (e.g. same_group, ar_punishment). Every per-node
        # tensor is flattened to (N, n_rounds) so edge_index gathers endpoints
        # directly; each is only passed when present, so empty edge_encoding
        # (punishment / legacy models) never requires any of them. Empty
        # edge_encoding -> (E, n_rounds, 0).
        edge_state = {}
        if "agent_group" in data:
            edge_state["agent_group"] = data["agent_group"].flatten(0, 1).to(device)
        y_masked_name = f"{self.y_name}_masked"
        if y_masked_name in data:
            edge_state[y_masked_name] = data[y_masked_name].flatten(0, 1).to(device)
        if "autoreg_mask" in data:
            edge_state["autoreg_mask"] = data["autoreg_mask"].flatten(0, 1).to(device)
        encoded["edge_attr"] = self.edge_encoder(
            edge_index=encoded["edge_index"], n_rounds=n_rounds, **edge_state
        )
        return encoded

    def predict_encoded(self, data, sample=True, reset_rnn=True):
        self.eval()
        y_logit = self(data, reset_rnn)
        y_pred_proba = th.nn.functional.softmax(y_logit, dim=-1)
        y_pred = self.y_encoder.decode(y_pred_proba, sample)
        return y_pred, y_pred_proba

    def predict_independent(self, data, sample=True, reset_rnn=True, edge_index=None):
        n_batch, n_nodes, n_rounds = data[self.y_name].shape
        if edge_index is None:
            edge_index = self.create_fully_connected(n_nodes, n_batch=n_batch)
        encoded = self.encode(
            data, y_encode=False, edge_index=edge_index, device=self.device
        )
        predict = self.predict_encoded(encoded, sample=sample, reset_rnn=reset_rnn)
        predict = tuple(t.reshape((n_batch, n_nodes, *t.shape[1:])) for t in predict)
        return predict

    def _copula_levels(self, proba_i, z, group_i):
        """Levels for one AR step from a shared severity latent.

        ``u = Phi(sqrt(rho) z_g + sqrt(1-rho) eps)`` inverted through the
        agent's own conditional CDF, so the AR marginals are preserved exactly
        and only the within-group, within-round dependence changes. Exactly one
        ``randn`` call per AR step regardless of group composition, so the RNG
        stream is composition-stable; ``eps`` is per (batch, round), ``z`` is
        drawn once per call and shared by every agent of a group-round.

        ``z`` carries a full node axis so a group id can index it directly --
        group ids are always < n_nodes, and reusing the node axis avoids
        renumbering groups (which would make the draw count composition
        dependent). ``proba_i``: (n_batch, n_rounds, y_levels); ``z``:
        (n_batch, n_nodes, n_rounds); ``group_i``: (n_batch, n_rounds) int64.
        Inverse-CDF convention: ``min{a : F(a) >= u}`` (``searchsorted`` with
        the default ``right=False``), matching the linear severity copula.
        """
        n_batch, n_rounds, _ = proba_i.shape
        assert int(group_i.max()) < z.shape[1], (
            f"agent_group id {int(group_i.max())} exceeds the latent node axis "
            f"({z.shape[1]}); group ids must index z directly"
        )
        eps = th.randn((n_batch, n_rounds), device=z.device, dtype=th.float64)
        a = float(np.sqrt(self.copula_rho))
        b = float(np.sqrt(1.0 - self.copula_rho))
        idx = group_i.long().unsqueeze(1)  # (n_batch, 1, n_rounds)
        z_g = z.gather(1, idx).squeeze(1)  # (n_batch, n_rounds)
        u = th.special.ndtr(a * z_g + b * eps)
        cum = proba_i.double().cumsum(-1)
        lvl = th.searchsorted(cum.contiguous(), u.unsqueeze(-1).contiguous())
        return lvl.squeeze(-1).clamp(0, self.y_levels - 1).to(th.int64)

    def predict_autoreg(self, data, sample=True, reset_rnn=True, edge_index=None):
        # `reset_rnn` accepted for signature parity with predict_independent.
        # `edge_index` honoured if provided so batched callers don't pay to
        # rebuild it per call.
        #
        # RNN contract: the full round history is re-fed on every AR step and
        # `self(encoded)` runs with `reset_rnn=True`, so the GRU advances once
        # per round and never per AR step. The reveal order comes from the
        # caller's seeded numpy RNG, and downstream `MultiManager` consumes
        # only round -1.
        self.eval()

        n_batch, n_nodes, n_rounds = data["contribution"].shape
        if edge_index is None:
            edge_index = self.create_fully_connected(n_nodes, n_batch=n_batch)

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

        # Copula sampling is off unless a rho is set and we are sampling; both
        # None and 0.0 are falsy, and sample=False always keeps the argmax.
        # Nothing above this point draws, so the legacy path below consumes the
        # unmodified RNG stream.
        use_copula = bool(sample and self.copula_rho)
        if use_copula:
            assert "agent_group" in data, (
                "copula sampling needs 'agent_group' in the data to share one "
                "severity latent per group and round"
            )
            # one latent per (batch, group, round), drawn once per call and
            # held fixed across the AR steps of the round
            z = th.randn(
                (n_batch, n_nodes, n_rounds), device=self.device, dtype=th.float64
            )
            agent_group = data["agent_group"].to(self.device)

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
            y_pred_proba_ = th.nn.functional.softmax(y_logit, dim=-1)
            if use_copula:
                # decode() is never called here: its categorical draw would
                # both override the copula level and consume RNG. Only agent
                # i's slice is read below, so the other slots stay at zero.
                y_pred_proba_ = y_pred_proba_.reshape(
                    n_batch, n_nodes, n_rounds, self.y_levels
                )
                y_pred_ = th.zeros(
                    (n_batch, n_nodes, n_rounds), device=self.device, dtype=th.int64
                )
                y_pred_[:, i] = self._copula_levels(
                    y_pred_proba_[:, i], z, agent_group[:, i]
                )
            else:
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
            "copula_rho",
            "x_encoding",
            "u_encoding",
            "edge_encoding",
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

    def create_fully_connected(self, n_nodes, n_batch=1):
        return th.tensor(
            [
                [i + k * n_nodes, j + k * n_nodes]
                for k in range(n_batch)
                for i in range(n_nodes)
                for j in range(n_nodes)
                if i != j
            ],
            device=self.device,
        ).T
