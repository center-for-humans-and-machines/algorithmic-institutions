import torch as th


class SinglerEncoder(th.nn.Module):
    def __init__(self, encoding, name, n_levels):
        super(SinglerEncoder, self).__init__()
        self.name = name
        if encoding == 'ordinal':
            assert n_levels is not None
            self.map = th.tensor(
                [[1]*i + [0]*(n_levels - i - 1)
                for i in range(n_levels)], dtype=th.float
            )
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
        enc = self.map[state[self.name]]
        return enc


class Encoder(th.nn.Module):
    def __init__(self, encodings):
        super(Encoder, self).__init__()
        self.encoder = th.nn.ModuleList([
            SinglerEncoder(**e)
            for e in encodings
        ])
        self.size = sum(e.size for e in self.encoder)

    def forward(self, **state):
        encoding = [
            e(**state)
            for e in self.encoder
        ]
        return th.cat(encoding, axis=-1)