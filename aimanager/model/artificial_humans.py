import torch as th
from aimanager.model.neural.mlp import MultiLayer


class ArtificialHuman(th.nn.Module):
    def __init__(self, *, n_contributions, n_punishments, y_encoding = 'ordinal',  model=None, **model_args):
        super(ArtificialHuman, self).__init__()
        if y_encoding == 'ordinal':
            raise NotImplementedError('Currently not supported.')
            output_size = (n_contributions - 1)
        elif y_encoding == 'onehot':
            output_size = n_contributions
        elif y_encoding == 'numeric':
            raise NotImplementedError('Currently not supported.')
            output_size = 1
        else:
            raise ValueError(f'Unkown y encoding {y_encoding}')
        input_size = n_contributions + n_punishments
    
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
        logit = self(**enc)
        # print(logit.shape)
        prob = th.nn.functional.softmax(logit, dim=-1)
        # print(prob.shape)
        action = th.multinomial(prob, 1).squeeze(-1)
        # print(action.shape)
        return action#logit.argmax(axis=-1)

    def predict(self, **state):
        self.model.eval()
        y_pred_logit = self(**state)
        if self.y_encoding == 'ordinal':
            raise NotImplementedError('Currently not supported.')
            # y_pred_proba = th.sigmoid(y_pred_logit).detach().cpu().numpy()
            # y_pred = ordinal_to_int(y_pred_proba)
            # y_pred_proba = np.concatenate([np.ones_like(y_pred_proba[:,[0]]), y_pred_proba[:,:]], axis=1)
        elif self.y_encoding == 'onehot': 
            y_pred_proba = th.nn.functional.softmax(y_pred_logit, dim=-1)
            y_pred = y_pred_proba.argmax(axis=-1)
        elif self.y_encoding == 'numeric': 
            raise NotImplementedError('Currently not supported.')
            # y_pred = th.sigmoid(y_pred_logit).detach().cpu().numpy()
            # # TODO: n_contributions is hardcoded here
            # y_pred = np.around(y_pred*21, decimals=0).astype(np.int64)
            # y_pred_proba = None
        else:
            raise ValueError(f'Unkown y encoding {self.y_encoding}')
        return y_pred, y_pred_proba

    def encode_x(self, prev_punishments, prev_contributions, **_):
        return {
            'ah_x_enc': th.cat([
                th.nn.functional.one_hot(prev_contributions, num_classes=self.n_contributions).float(),
                th.nn.functional.one_hot(prev_punishments, num_classes=self.n_punishments).float()
            ], dim=-1)
        }

    def encode_y(self, contributions, **_):
        return {
            'ah_y_enc': th.nn.functional.one_hot(contributions, num_classes=self.n_contributions).float()
        }

    def get_lossfn(self):
        if self.y_encoding == 'ordinal':
            raise NotImplementedError()
            loss_fn = th.nn.BCEWithLogitsLoss()
        elif self.y_encoding == 'onehot':
            loss_fn = th.nn.CrossEntropyLoss(reduction='none')
        elif self.y_encoding == 'numeric':
            raise NotImplementedError()
            mse = th.nn.MSELoss()
            sig = th.nn.Sigmoid()
            def _loss_fn(yhat,y):
                yhat = sig(yhat)*self.n_contributions
                return mse(yhat[:,0],y)
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
