import torch as th


class RandomArtificialHumans(th.nn.Module):
    def __init__(self, *, device, max_contribution):
        super(RandomArtificialHumans, self).__init__()
        self.max_contribution = max_contribution
        self.device = device

    def forward(self, punishments, **kwargs):
        """
            view: batch (b), round (r), agents (a), inputs (i)
        """
        n_batch, n_rounds, n_agents = punishments.shape

        q = th.rand((n_batch, n_rounds, n_agents,
                    self.max_contribution), device=self.device)
        return q

    def act(self, **view):
        view = {
            k: v.unsqueeze(0).unsqueeze(0)
            for k, v in view.items()
        }
        q = self(**view)
        q = q[0, 0]
        return q.argmax(dim=1)
