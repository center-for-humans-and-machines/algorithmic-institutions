import torch as th
from torch_scatter import scatter_mean


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
            self.size = self.map.shape[-1]
        elif encoding == 'onehot':
            assert n_levels is not None
            self.map = th.tensor(
                [[0]*i + [1] + [0]*(n_levels - i - 1)
                 for i in range(n_levels)], dtype=th.float
            )
            self.size = self.map.shape[-1]
        elif encoding == 'numeric':
            self.map = th.linspace(0, 1, n_levels, dtype=th.float).unsqueeze(-1)
            self.size = self.map.shape[-1]
        elif encoding == 'projection':
            self.map = th.nn.Embedding(n_levels, 1)
            self.size = 1

    def forward(self, **state):
        tensor = state[self.name]
        if tensor.dtype == th.bool:
            tensor = tensor.type(th.int64)
        assert tensor.dtype == th.int64
        self.map = self.map.to(tensor.device)
        if self.encoding == 'projection':
            tensor = tensor % self.n_levels
            return self.map(tensor)
        else:
            return self.map[tensor]

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
        elif self.encoding == 'projection':
            raise NotImplementedError()


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


# class EmbeddingEncoder(th.nn.Module):
#     def __init__(self, name, dimensions, n_levels):
#         super(EmbeddingEncoder, self).__init__()
#         self.embedding = th.ones((n_levels, dimensions), dtype=th.float)
#         self.name = name
#         self.size = dimensions

#     def forward(self, **state):
#         return self.embedding[state[self.name]]


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

    def forward(self, datashape='batch_agent_round', **state):
        encoding = [
            e(**state)
            for e in self.encoder
        ]
        if len(self.encoder) >= 1:
            encoding = th.cat(encoding, axis=-1)
            if self.aggregation == 'mean':
                if datashape == 'batch_agent_round':
                    assert len(
                        encoding.shape) == 4, f'Expected 4 dimensions. Shape is {len(encoding.shape)}'
                    encoding = encoding.mean(dim=1, keepdim=self.keepdim)
                elif datashape == 'batch*agent_round':
                    assert len(
                        encoding.shape) == 3, f'Expected 3 dimensions. Shape is {len(encoding.shape)}'
                    encoding = scatter_mean(encoding, state['batch'], dim=0)
                else:
                    raise ValueError('Unknown datashape.')
            else:
                assert self.aggregation is None, f"Unknown aggregation type {self.aggregation}"

        else:
            if self.aggregation == 'mean':
                if datashape == 'batch_agent_round':
                    assert len(
                        state[self.refrence].shape) == 3, f'Expected 3 dimensions. Shape is {len(state[self.refrence].shape)}'
                    enc_shape = (state[self.refrence].shape[0], 1,
                                 state[self.refrence].shape[2], 0)
                elif datashape == 'batch*agent_round':
                    assert len(
                        state[self.refrence].shape) == 2, f'Expected 2 dimensions. Shape is {len(state[self.refrence].shape)}'
                    enc_shape = (state['batch'].max()+1, state[self.refrence].shape[1], 0)
                else:
                    raise ValueError('Unknown datashape.')
            else:
                assert self.aggregation is None, f"Unknown aggregation type {self.aggregation}"
                enc_shape = state[self.refrence].shape + (0,)
            encoding = th.empty(enc_shape, device=state[self.refrence].device)
        return encoding
