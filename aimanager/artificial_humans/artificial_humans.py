import torch as th
from zmq import device
from aimanager.generic.mlp import MultiLayer
from aimanager.generic.encoder import Encoder, IntEncoder

def ordinal_to_int(arr):
    """
    Get the position of the first 0
    """
    n_levels = arr.shape[-1] + 1
    integers = th.arange(n_levels-1, 0, -1)
    arr = (arr < 0.5).float()
    arr = arr * integers.unsqueeze(-1)
    return n_levels - arr.max(1) - 1



class ArtificialHuman(th.nn.Module):
    def __init__(self, *, n_contributions, n_punishments, y_encoding = 'ordinal', x_encoding=None,  model=None, **model_args):
        super(ArtificialHuman, self).__init__()
        if y_encoding == 'ordinal':
            output_size = (n_contributions - 1)
        elif y_encoding == 'onehot':
            output_size = n_contributions
        elif y_encoding == 'numeric':
            output_size = 1
        else:
            raise ValueError(f'Unkown y encoding {y_encoding}')

        self.x_encoder = Encoder(x_encoding)
        self.y_encoder = IntEncoder(encoding=y_encoding, name='contributions', n_levels=n_contributions)

        input_size = self.x_encoder.size
    
        if not model:
            self.model = MultiLayer(output_size=output_size, input_size=input_size, **model_args)
        else:
            self.model = model
        self.y_encoding = y_encoding
        self.n_contributions = n_contributions
        self.n_punishments = n_punishments

    def forward(self, ah_x_enc, **_):
        """
        Takes an already encoded tensor
        """
        return self.model(ah_x_enc)
        
    def act(self, **state):
        enc = self.encode_x(**state)
        pred, prob = self.predict(**enc)
        action = th.multinomial(prob, 1).squeeze(-1)
        return action

    def predict(self, **encoding):
        self.model.eval()
        y_pred_logit = self(**encoding)
        if self.y_encoding == 'ordinal':
            y_pred_proba = th.sigmoid(y_pred_logit)
            y_pred = ordinal_to_int(y_pred_proba)
            y_pred_proba = th.cat([th.ones_like(y_pred_proba[:,[0]]), y_pred_proba], axis=1)
            y_pred_proba = y_pred_proba / th.sum(y_pred_proba, dim=-1, keepdim=True)
        elif self.y_encoding == 'onehot': 
            y_pred_proba = th.nn.functional.softmax(y_pred_logit, dim=-1)
            y_pred = y_pred_proba.argmax(axis=-1)
        elif self.y_encoding == 'numeric':
            y_pred = th.sigmoid(y_pred_logit.squeeze(-1))
            y_pred = th.round(y_pred * self.n_contributions)
            y_pred = y_pred.type(th.int64)
            y_pred_proba = th.nn.functional.one_hot(y_pred, num_classes=self.n_contributions).float()
        else:
            raise ValueError(f'Unkown y encoding {self.y_encoding}')
        return y_pred, y_pred_proba

    def encode_x(self, **data):
        return {
            'ah_x_enc': self.x_encoder(**data)
        }

    def encode_y(self, **state):
        return {
            'ah_y_enc': self.y_encoder(**state)
        }

    def get_lossfn(self):
        if self.y_encoding == 'ordinal':
            loss_fn = th.nn.BCEWithLogitsLoss()
        elif self.y_encoding == 'onehot':
            loss_fn = th.nn.CrossEntropyLoss(reduction='none')
        elif self.y_encoding == 'numeric':
            mse = th.nn.MSELoss()
            sig = th.nn.Sigmoid()
            def _loss_fn(yhat,y):
                yhat = sig(yhat.squeeze(-1))*self.n_contributions
                return mse(yhat,y.squeeze(-1))
            loss_fn = _loss_fn
        return loss_fn

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
