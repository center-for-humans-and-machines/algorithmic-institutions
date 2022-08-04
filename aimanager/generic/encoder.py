import torch as th


class IntEncoder(th.nn.Module):
    def __init__(self, encoding, name, n_levels):
        super(IntEncoder, self).__init__()
        self.name = name
        self.encoding = encoding
        self.n_levels = n_levels
        if encoding == 'ordinal':
            assert n_levels is not None
            self.map = th.tensor(
                [[1]*i + [0]*(n_levels - i - 1)
                 for i in range(n_levels)], dtype=th.float
            )
            self.position_values = th.arange(n_levels-1, 0, -1, dtype=th.float)
        elif encoding == 'onehot':
            assert n_levels is not None
            self.map = th.tensor(
                [[0]*i + [1] + [0]*(n_levels - i - 1)
                 for i in range(n_levels)], dtype=th.float
            )
        elif encoding == 'numeric':
            self.map = th.linspace(0, 1, n_levels, dtype=th.float).unsqueeze(-1)
        self.size = self.map.shape[-1]

    def forward(self, **state):
        assert state[self.name].dtype == th.int64
        if self.map.device != state[self.name].device:
            self.map = self.map.to(state[self.name].device)
        enc = self.map[state[self.name]]
        return enc

    def decode(self, arr, sample=False):
        if self.encoding == 'ordinal':
            raise NotImplementedError()
            arr = (arr < 0.5).float()
            arr = th.einsum('ijkl,l->ijkl', arr, self.position_values)
            return self.n_levels - arr.max(-1)[0] - 1
        elif self.encoding == 'onehot':
            if sample:
                dec = th.multinomial(arr.reshape(-1, arr.shape[-1]), 1)
                return dec.reshape(arr.shape[:-1])
            else:
                return arr.argmax(axis=-1)
        elif self.encoding == 'numeric':
            arr = th.round(arr * (self.n_levels - 1))
            return arr.type(th.int64)


class FloatEncoder(th.nn.Module):
    def __init__(self, norm, name):
        super(FloatEncoder, self).__init__()
        self.size = 1
        self.norm = norm
        self.name = name

    def forward(self, **state):
        assert state[self.name].dtype == th.float
        enc = (state[self.name] / self.norm).unsqueeze(-1)
        return enc


class BoolEncoder(th.nn.Module):
    def __init__(self, name):
        super(BoolEncoder, self).__init__()
        self.size = 1
        self.name = name

    def forward(self, **state):
        assert state[self.name].dtype == th.bool
        enc = state[self.name].float().unsqueeze(-1)
        return enc


encoder = {
    'int': IntEncoder,
    'float': FloatEncoder,
    'bool': BoolEncoder
}


def get_encoder(etype='int', **kwargs):
    return encoder[etype](**kwargs)


class Encoder(th.nn.Module):
    def __init__(self, encodings, aggregation=None, keepdim=True, refrence=None):
        super(Encoder, self).__init__()
        self.encoder = th.nn.ModuleList([
            get_encoder(**e)
            for e in encodings
        ])
        self.size = sum(e.size for e in self.encoder)
        self.aggregation = aggregation
        self.keepdim = keepdim
        self.refrence = refrence

    def forward(self, **state):
        encoding = [
            e(**state)
            for e in self.encoder
        ]
        if len(self.encoder) >= 1:
            encoding = th.cat(encoding, axis=-1)
        else:
            encoding = th.empty(state[self.refrence].shape + (0,),
                                device=state[self.refrence].device)

        if self.aggregation == 'mean':
            encoding = encoding.mean(dim=1, keepdim=self.keepdim)
        else:
            assert self.aggregation is None, f"Unknown aggregation type {self.aggregation}"
        return encoding
