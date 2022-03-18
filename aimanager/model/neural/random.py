import torch as th


class RandomArtificialHumans(th.nn.Module):
    def __init__(self, *, device, max_contribution):
        super(RandomArtificialHumans, self).__init__()
        self.max_contribution = max_contribution
        self.device = device
    
    def forward(self, view):
        """
            view: batch (b), round (r), agents (a), inputs (i)
        """
        n_batch, n_rounds, n_agents, n_inputs  = view.shape 

        q = th.rand((n_batch, n_rounds, n_agents, self.max_contribution), device=self.device)
        return q