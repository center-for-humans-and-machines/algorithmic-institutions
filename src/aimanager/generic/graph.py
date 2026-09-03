from torch.nn import Sequential as Seq, Linear as Lin, Tanh, GRU
import numpy as np
import torch as th
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
from aimanager.generic.conditional_bernoulli import sample_conditional_bernoulli
from aimanager.generic.copula import sample_correlated_levels
from aimanager.generic.encoder import Encoder, IntEncoder
from aimanager.generic.joint_exodus import JointExodusHead


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


EDGE_ENCODERS = {"same_group": SameGroupEdgeEncoder}


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
        x_encoding=[],
        u_encoding=[],
        edge_encoding=[],
        add_rnn=True,
        add_edge_model=True,
        add_global_model=True,
        hidden_size=None,
        default_values={},
        copula_rho=0.0,
        copula_phi=0.0,
        copula_switch_every=None,
        joint_exodus=False,
        joint_exodus_head=None,
        joint_exodus_switch_every=None,
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

        # Herding copula, defined for the switch and contribution heads;
        # see notes/autoresearch_log/switch-herding-copula.md and
        # notes/autoresearch_log/contribution-herding-copula-v2.md.
        # rho 0 keeps the legacy independent draw; phi 1 is the unit-root
        # boundary, a latent held static for the whole episode.
        assert 0.0 <= copula_rho < 1.0, f"copula_rho must be in [0, 1), {copula_rho}"
        assert 0.0 <= copula_phi <= 1.0, f"copula_phi must be in [0, 1], {copula_phi}"
        assert copula_rho == 0.0 or y_name in (
            "does_switch",
            "contribution",
        ), (
            "copula_rho > 0 is only defined for does_switch and contribution, "
            f"got {y_name}"
        )
        assert copula_phi == 0.0 or (
            isinstance(copula_switch_every, int) and copula_switch_every > 0
        ), "copula_phi > 0 requires a positive int copula_switch_every"
        self.copula_rho = copula_rho
        self.copula_phi = copula_phi
        self.copula_switch_every = copula_switch_every
        self._copula_z = None

        # Joint exodus head, a round-level joint over the pair of leaver
        # counts (m_0, m_1); see notes/autoresearch_log/switch-joint-exodus.md
        # and generic/joint_exodus.py. Off by default: an artifact saved
        # without these two keys loads with the head absent and behaves
        # exactly as it does today.
        assert isinstance(
            joint_exodus, bool
        ), f"joint_exodus must be a bool, got {joint_exodus!r}"
        assert not joint_exodus or y_name == "does_switch", (
            "joint_exodus is only defined for the does_switch head, " f"got {y_name}"
        )
        # Two different switch samplers; enabling both would leave the draw
        # decided by nothing but the order of the dispatch below.
        assert not (joint_exodus and copula_rho > 0), (
            "joint_exodus and the herding copula are alternative switch "
            "samplers -- enable one or the other, not both"
        )
        # Which rounds the joint draw fires on (plan step 4). The predictor is
        # run EVERY round to keep its GRU warm but its output is only consumed
        # on decision rounds, so the joint machinery -- and the RNG it eats --
        # must be confined to those rounds. Same predicate and same role as
        # `copula_switch_every`: decision rounds are `(r + 1) % every == 0`.
        # `bool` is excluded explicitly: `True` is an `int` and would silently
        # mean "every round is a decision round".
        assert joint_exodus_switch_every is None or (
            isinstance(joint_exodus_switch_every, int)
            and not isinstance(joint_exodus_switch_every, bool)
            and joint_exodus_switch_every > 0
        ), (
            "joint_exodus_switch_every must be None or a positive int, got "
            f"{joint_exodus_switch_every!r}"
        )
        assert joint_exodus or joint_exodus_switch_every is None, (
            "joint_exodus_switch_every is only meaningful with the joint "
            "exodus head enabled"
        )
        self.joint_exodus = joint_exodus
        self.joint_exodus_switch_every = joint_exodus_switch_every

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

            # Built LAST, so every pre-existing parameter is initialised from
            # exactly the RNG state it saw before this head existed: a model
            # with the head off is bit-identical to today, and a model with it
            # on shares the same trunk weights under the same seed.
            # `x_features` is the post-RNN node width the head reads.
            if joint_exodus_head is not None:
                self.joint_exodus_head = joint_exodus_head
            elif joint_exodus:
                self.joint_exodus_head = JointExodusHead(
                    embed_size=x_features, hidden_size=hidden_size
                )
            else:
                self.joint_exodus_head = None

        else:
            self.op1 = op1
            self.op2 = op2
            self.rnn_n = rnn_n
            self.rnn_g = rnn_g
            self.bias = bias
            self.rnn_n_h0 = None
            self.rnn_g_h0 = None
            self.joint_exodus_head = joint_exodus_head

        assert (self.joint_exodus_head is not None) == self.joint_exodus, (
            "joint_exodus and joint_exodus_head disagree: "
            f"{self.joint_exodus} vs {type(self.joint_exodus_head).__name__}"
        )

    def forward(self, data, reset_rnn=True, return_joint=False, decider_mask=None):
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
        # The joint exodus head branches off the post-RNN node embeddings and
        # feeds nothing back, so the per-agent path below is untouched whether
        # the head runs or not. It feeds nothing back on the BACKWARD pass
        # either: the head detaches the pooled embeddings it reads (plan step
        # 2b, see `joint_exodus.JointExodusHead.forward`), so the joint loss
        # leaves every parameter above this line -- op1, the RNNs, the
        # encoders -- with exactly the gradient it would have had with the
        # head off. `return_joint` defaults to False, so every existing call
        # site keeps the exact signature and return type it has today.
        # `_predict_encoded_joint_exodus` is the one caller that asks for it.
        joint = None
        if return_joint and self.joint_exodus_head is not None:
            joint = self.joint_exodus_head(
                x,
                agent_group=data["agent_group"],
                round_number=data["round_number"],
                batch=batch,
                decider_mask=data.get("mask") if decider_mask is None else decider_mask,
            )
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
        if return_joint:
            return x, joint
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
        # Per-edge features (e.g. same_group). agent_group is flattened to
        # (N, n_rounds) so edge_index gathers endpoints directly; only passed
        # when present, so empty edge_encoding (punishment / legacy models)
        # never requires it. Empty edge_encoding -> (E, n_rounds, 0).
        edge_state = {}
        if "agent_group" in data:
            edge_state["agent_group"] = data["agent_group"].flatten(0, 1).to(device)
        encoded["edge_attr"] = self.edge_encoder(
            edge_index=encoded["edge_index"], n_rounds=n_rounds, **edge_state
        )
        # The joint exodus head pools by membership and conditions on the
        # round, neither of which the per-agent path needs, so both are only
        # carried when the head is present -- a model without it encodes
        # exactly the keys it encodes today.
        if self.joint_exodus_head is not None:
            for key in ("agent_group", "round_number"):
                assert key in data, f"the joint exodus head requires {key} in the state"
                encoded[key] = data[key].flatten(0, 1).to(device)
        return encoded

    def predict_encoded(self, data, sample=True, reset_rnn=True):
        self.eval()
        y_logit = self(data, reset_rnn)
        y_pred_proba = th.nn.functional.softmax(y_logit, dim=-1)
        y_pred = self.y_encoder.decode(y_pred_proba, sample)
        return y_pred, y_pred_proba

    def _predict_encoded_copula(self, data, encoded, shape, reset_rnn=True):
        """Same forward as `predict_encoded`, but the level is drawn with a
        shared latent per (batch, agent_group) cell -- the herding sampler of
        notes/autoresearch_log/switch-herding-copula.md."""
        n_batch, n_nodes, n_rounds = shape
        self.eval()
        y_logit = self(encoded, reset_rnn)
        y_pred_proba = th.nn.functional.softmax(y_logit, dim=-1)
        if reset_rnn:
            self._copula_z = None
        # The sampler draws on the cpu RNG; agent_group is 0/1, so the dense
        # cell id of the round being sampled is batch_index * 2 + agent_group.
        agent_group = data["agent_group"].reshape(n_batch, n_nodes, n_rounds).cpu()
        cells = th.arange(n_batch).reshape(-1, 1, 1) * 2 + agent_group
        proba = y_pred_proba.detach().cpu()
        y_pred = th.empty(proba.shape[:-1], dtype=th.int64)
        keep_z = self.copula_phi > 0
        rounds = (
            data["round_number"].reshape(n_batch, n_nodes, n_rounds)[0, 0]
            if keep_z
            else None
        )
        for r in range(n_rounds):
            cell_id = cells[:, :, r].reshape(-1)
            n_cells = int(cell_id.max()) + 1
            z_prev = None if self._copula_z is None else self._copula_z[:n_cells]
            levels, z_cell = sample_correlated_levels(
                proba[:, r],
                cell_id,
                self.copula_rho,
                z_prev=z_prev,
                phi=self.copula_phi,
            )
            y_pred[:, r] = levels
            # Only a decision round advances the AR(1) latent (ruling D6): the
            # predictor runs every round, so advancing per call would decay the
            # realized persistence to phi ** copula_switch_every. The store is
            # kept at full length because a group can empty out (then n_cells
            # falls below 2 * n_batch), and the cell index must stay stable.
            if keep_z and (int(rounds[r]) + 1) % self.copula_switch_every == 0:
                z_store = (
                    th.zeros(2 * n_batch, dtype=z_cell.dtype)
                    if self._copula_z is None
                    else self._copula_z.clone()
                )
                z_store[:n_cells] = z_cell
                self._copula_z = z_store
        return y_pred.to(y_pred_proba.device), y_pred_proba

    def _predict_encoded_joint_exodus(self, encoded, shape, reset_rnn=True):
        """Same forward as `predict_encoded`, but on a DECISION round the
        independent per-agent draw is replaced by the two-stage joint draw of
        notes/autoresearch_log/switch-joint-exodus.md (plan step 4):

        1. per-agent switch probabilities from the existing, UNCHANGED
           per-agent head -- this method never touches those logits;
        2. a pair `(m_0, m_1)` of leaver counts drawn per batch element from
           the joint head's masked joint over the padded 9 x 9 grid;
        3. WHICH members leave, drawn per group by the exact conditional
           Bernoulli of `generic/conditional_bernoulli.py`, conditioned on the
           very probabilities from 1.

        RNG. The switch predictor runs every round to keep its GRU hidden
        state warm, but its output is only consumed on decision rounds
        (`manager/environment.py: step`). So a NON-decision round must leave
        the global RNG exactly where the independent path would have: the
        legacy `y_encoder.decode(..., sample=True)` below is taken verbatim
        and, off a decision round, is the only draw this method makes. On a
        decision round that legacy draw is made and then DISCARDED -- one
        wasted categorical per decision round buys the guarantee that the
        neutral path is the pre-existing expression itself rather than a
        re-derivation of it, and the mechanism is meant to differ on exactly
        those rounds anyway.

        Everything stays on the model's device, like the legacy draw and
        unlike `_predict_encoded_copula` (whose latent algebra is float64 on
        the cpu RNG by construction).
        """
        n_batch, n_nodes, n_rounds = shape
        head = self.joint_exodus_head
        assert head is not None, "no joint exodus head to sample from"
        assert self.joint_exodus_switch_every is not None, (
            "the joint exodus head is present but joint_exodus_switch_every "
            "is not set, so the decision rounds it must fire on are unknown. "
            "Set it on the model (see configs/training/artificial_humans/"
            "switch_predictor/joint_exodus.yml) rather than sampling the "
            "per-agent head and silently reporting a joint-exodus run."
        )
        assert self.y_levels == 2, (
            "the joint exodus draw is a leaver COUNT over a binary label, "
            f"got y_levels={self.y_levels}"
        )
        assert n_nodes <= head.max_group_size, (
            f"{n_nodes} players do not fit the head's count grid "
            f"(max_group_size={head.max_group_size})"
        )
        self.eval()
        y_logit, joint = self(encoded, reset_rnn, True)
        y_pred_proba = th.nn.functional.softmax(y_logit, dim=-1)

        # (1) The per-agent draw, verbatim. Off a decision round this is the
        # whole method; see the RNG note above.
        y_pred = self.y_encoder.decode(y_pred_proba, True)

        log_prob, k = joint
        agent_group = encoded["agent_group"].reshape(n_batch, n_nodes, n_rounds)
        round_number = encoded["round_number"].reshape(n_batch, n_nodes, n_rounds)
        # The gate is read off one node; a round that differed across the
        # batch would gate some episodes on another's clock.
        assert bool(
            (round_number == round_number[0, 0].reshape(1, 1, n_rounds)).all()
        ), "round_number is not constant within a round across the batch"
        rounds = round_number[0, 0]
        grid = log_prob.shape[-1]

        for r in range(n_rounds):
            if (int(rounds[r]) + 1) % self.joint_exodus_switch_every != 0:
                continue

            # (2) the pair (m_0, m_1), one draw for the whole batch. The grid
            # is already masked to m_g <= k_g and renormalised, so the pair is
            # feasible by construction and (0, 0) is always available -- a
            # fully merged round (k = 8, k = 0) is an ordinary cell here.
            pair_proba = log_prob[:, r].exp().flatten(-2, -1)
            cell = th.multinomial(pair_proba, 1).reshape(-1)  # draw 1/1
            # Row-major over (m_0, m_1), the same flattening the training loss
            # gathers with (`train.joint_exodus_loss`: m_0 * grid + m_1).
            m = th.stack(
                [th.div(cell, grid, rounding_mode="floor"), cell % grid], dim=-1
            )  # (n_batch, 2)

            # (3) which members leave. Membership is read at the moment of the
            # decision, PRE-switch, from the same `agent_group` the head
            # pooled with -- so the count drawn for group g is spent on group
            # g's own members. The identity below is the guard against that
            # ever silently inverting.
            group_r = agent_group[:, :, r]
            member = th.stack(
                [group_r == g for g in range(head.n_groups)], dim=0
            )  # (n_groups, n_batch, n_nodes)
            assert th.equal(member.sum(-1).transpose(0, 1), k[:, r]), (
                "the per-group membership used to place the leavers and the "
                "head's own valid-decider count disagree"
            )
            # One batched conditional-Bernoulli call over both groups, so the
            # RNG cost is one categorical draw and not one per group. Rows are
            # group-major: [group 0 of every episode, group 1 of every
            # episode], and `p` is tiled to match. Every row is n_nodes wide
            # with the non-members masked out, so a selected slot IS the agent
            # index and no gather can transpose the two groups.
            p_switch = y_pred_proba[:, r, -1].reshape(n_batch, n_nodes)
            selected = sample_conditional_bernoulli(
                p_switch.repeat(head.n_groups, 1),
                m.transpose(0, 1).reshape(-1),
                mask=member.reshape(-1, n_nodes),
                max_group_size=n_nodes,
            )
            # The two masks are disjoint and cover every agent exactly once,
            # so the union is the round's switch vector.
            switch = selected.reshape(head.n_groups, n_batch, n_nodes).any(0)
            y_pred[:, r] = switch.reshape(-1).to(y_pred.dtype)

        return y_pred, y_pred_proba

    def predict_independent(self, data, sample=True, reset_rnn=True, edge_index=None):
        n_batch, n_nodes, n_rounds = data[self.y_name].shape
        if edge_index is None:
            edge_index = self.create_fully_connected(n_nodes, n_batch=n_batch)
        encoded = self.encode(
            data, y_encode=False, edge_index=edge_index, device=self.device
        )
        if sample and self.joint_exodus_head is not None:
            predict = self._predict_encoded_joint_exodus(
                encoded, (n_batch, n_nodes, n_rounds), reset_rnn=reset_rnn
            )
        elif sample and self.copula_rho > 0:
            predict = self._predict_encoded_copula(
                data, encoded, (n_batch, n_nodes, n_rounds), reset_rnn=reset_rnn
            )
        else:
            predict = self.predict_encoded(encoded, sample=sample, reset_rnn=reset_rnn)
        predict = tuple(t.reshape((n_batch, n_nodes, *t.shape[1:])) for t in predict)
        return predict

    def predict_autoreg(self, data, sample=True, reset_rnn=True, edge_index=None):
        # `reset_rnn` accepted for signature parity with predict_independent
        # (the autoreg model has no RNN). `edge_index` honoured if provided so
        # batched callers don't pay to rebuild it per call.
        self.eval()
        assert (
            self.rnn_n is None and self.rnn_g is None
        ), "Autoregressive predictions do not support RNN"

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
            "edge_encoding",
            "b_encoding",
            "default_values",
            "copula_rho",
            "copula_phi",
            "copula_switch_every",
            "joint_exodus",
            "joint_exodus_head",
            "joint_exodus_switch_every",
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
