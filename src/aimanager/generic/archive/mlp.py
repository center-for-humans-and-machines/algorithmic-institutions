import torch as th
from typing import Literal, Optional


class FeedForwardLayer(th.nn.Module):
    def __init__(
            self, *,
            input_size: int, 
            hidden_size: int, 
            dropout: Optional[float], 
            activation: Optional[Literal['relu', 'logit', 'softmax']]):
        super(FeedForwardLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.lin = th.nn.Linear(self.input_size, self.hidden_size)

        if activation == 'relu':
            self.activation = th.nn.ReLU()
        elif activation == 'logit':
            self.activation = th.nn.Logit()
        elif activation == 'softmax':
            self.activation = th.nn.Softmax()
        else:
            self.activation = None

        if dropout:
            self.dropout = th.nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.lin(x)
        if self.dropout:
            x = self.dropout(x)
        if self.activation:
            x = self.activation(x)
        return x


class MultiLayer(th.nn.Module):
    def __init__(self, *, 
            n_layers: int, 
            hidden_size: Optional[int]=None, 
            input_size: int, 
            output_size: int, 
            dropout: Optional[float]=None):
        super(MultiLayer, self).__init__()
        
        assert not ((hidden_size == None) and (n_layers > 1))

        self.layers = th.nn.Sequential(
            *(FeedForwardLayer(
                input_size=hidden_size if i > 0 else input_size,
                hidden_size=output_size if i == (n_layers - 1) else hidden_size,
                dropout=dropout,
                activation=None if i == (n_layers - 1) else 'relu'
            )
            for i in range(n_layers))
        )
            
    def forward(self, x):
        return self.layers(x)
