import torch as th
from aimanager.generic.mlp import MultiLayer
from aimanager.generic.encoder import Encoder, IntEncoder
from torch_geometric.loader import DataLoader

class MLPArtificialHuman(th.nn.Module):
    def __init__(self, *, n_contributions, n_punishments, x_encoding=None,  model=None, **model_args):
        super(MLPArtificialHuman, self).__init__()
        output_size = n_contributions
        self.y_encoding = 'onehot'

        self.x_encoder = Encoder(x_encoding)
        self.y_encoder = IntEncoder(encoding='onehot', name='contributions', n_levels=n_contributions)

        input_size = self.x_encoder.size
    
        if not model:
            self.model = MultiLayer(output_size=output_size, input_size=input_size, **model_args)
        else:
            self.model = model
        self.n_contributions = n_contributions
        self.n_punishments = n_punishments


    def forward(self, data):
        """
        Takes an already encoded tensor
        """
        return self.model(data['x'])
        
    def act(self, **state):
        raise NotImplementedError('have to fix this')
        enc = self.encode_x(**state)
        pred, prob = self.predict(**enc)
        action = th.multinomial(prob, 1).squeeze(-1)
        return action

    def predict(self, data):
        self.model.eval()
        y_pred_logit = th.cat([self(d)
            for d in iter(DataLoader(data, shuffle=False, batch_size=10))
        ])
        y_pred_proba = th.nn.functional.softmax(y_pred_logit, dim=-1)
        y_pred = self.y_encoder.decode(y_pred_proba)
        return y_pred, y_pred_proba

    def save(self, filename):
        to_save = {
            'model': self.model,
            'y_encoding': self.y_encoding,
            'n_contributions': self.n_contributions,
            'n_punishments': self.n_punishments
        }
        th.save(to_save, filename)

    @classmethod
    def load(cls, filename):
        to_load = th.load(filename)
        ah = cls(**to_load)
        return ah
