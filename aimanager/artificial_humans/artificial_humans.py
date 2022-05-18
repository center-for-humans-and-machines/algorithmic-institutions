import torch as th
from aimanager.generic.mlp import MultiLayer
from aimanager.generic.encoder import Encoder, IntEncoder


class ArtificialHuman(th.nn.Module):
    def __init__(self, *, n_contributions, n_punishments, y_encoding='ordinal', y_scaling='sigmoid', x_encoding=None,  model=None, **model_args):
        super(ArtificialHuman, self).__init__()
        if y_encoding == 'ordinal':
            raise NotImplementedError()
            output_size = (n_contributions - 1)
        elif y_encoding == 'onehot':
            output_size = n_contributions
        elif y_encoding == 'numeric':
            output_size = 1
        else:
            raise ValueError(f'Unkown y encoding {y_encoding}')

        print(y_scaling)

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
        self.y_scaling = y_scaling

    def forward(self, ah_x_enc, **_):
        """
        Takes an already encoded tensor
        """
        return self.model(ah_x_enc)
        
    def act(self, **state):
        raise NotImplementedError('have to fix this')
        enc = self.encode_x(**state)
        pred, prob = self.predict(**enc)
        action = th.multinomial(prob, 1).squeeze(-1)
        return action

    def predict(self, **encoding):
        self.model.eval()
        y_pred_logit = self(**encoding)
        if self.y_encoding == 'ordinal':
            raise NotImplementedError()
            y_pred_proba = th.sigmoid(y_pred_logit)
            y_pred = self.y_encoder.decode(y_pred_proba)
            y_pred_proba = None
            # y_pred_proba = th.cat([th.ones((*y_pred_proba.shape[:-1],1)), y_pred_proba], axis=-1)
            # y_pred_proba = y_pred_proba / th.sum(y_pred_proba, dim=-1, keepdim=True)
        elif self.y_encoding == 'onehot': 
            y_pred_proba = th.nn.functional.softmax(y_pred_logit, dim=-1)
            y_pred = self.y_encoder.decode(y_pred_proba)
        elif self.y_encoding == 'numeric':
            if self.y_scaling == 'sigmoid':
                y_pred = th.sigmoid(y_pred_logit.squeeze(-1))
            elif self.y_scaling in ['hardtanh','None']:
                y_pred = th.nn.functional.hardtanh(y_pred_logit, min_val=0.0, max_val=1.0).squeeze(-1)
            else:
                raise ValueError('Unkown y scaling.')
            y_pred = self.y_encoder.decode(y_pred)
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
            def _loss_fn(yhat,y):
                if self.y_scaling == 'sigmoid':
                    yhat = th.sigmoid(yhat)
                elif self.y_scaling == 'hardtanh':
                    yhat = th.nn.functional.hardtanh(yhat, min_val=0.0, max_val=1.0)
                elif self.y_scaling == 'None':
                    pass
                else:
                    raise ValueError('Unkown y_scaling.')
                return mse(yhat,y)
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
