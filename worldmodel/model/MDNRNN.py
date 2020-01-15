import torch
import torch.nn as nn
import torch.nn.functional as F

class MDNRNN(nn.Module):
    def __init__(self, z_size, n_hidden, n_gaussians=5, n_layers=1):
        super(MDNRNN, self).__init__()

        self.z_size = z_size
        self.n_hidden = n_hidden
        self.n_gaussians = n_gaussians
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(z_size, n_hidden, n_layers, batch_first=True)
        self.fc1 = nn.Linear(n_hidden, n_gaussians*z_size)
        self.fc2 = nn.Linear(n_hidden, n_gaussians*z_size)
        self.fc3 = nn.Linear(n_hidden, n_gaussians*z_size)
        
    def get_mixture_coef(self, y):
        rollout_length = y.size(1)
        pi, mu, sigma = self.fc1(y), self.fc2(y), self.fc3(y)
        
        pi = pi.view(-1, rollout_length, self.n_gaussians, self.z_size)
        mu = mu.view(-1, rollout_length, self.n_gaussians, self.z_size)
        sigma = sigma.view(-1, rollout_length, self.n_gaussians, self.z_size)
        
        pi = F.softmax(pi, 2)
        sigma = torch.exp(sigma)
        return pi, mu, sigma  
        
    def forward(self, x, h):
        y, (h, c) = self.lstm(x, h)
        pi, mu, sigma = self.get_mixture_coef(y)
        return (pi, mu, sigma), (h, c)
    
    def init_hidden(self, bsz, device):
        return (torch.zeros(self.n_layers, bsz, self.n_hidden).to(device),
                torch.zeros(self.n_layers, bsz, self.n_hidden).to(device))

    def play_encode(self, x, h):
        with torch.no_grad():
            return self.forward(x, h)

    @classmethod
    def load_model(cls, path, *args, **kwargs):
        state_dict = torch.load(path, map_location='cpu')
        rnn = cls(*args, **kwargs)
        rnn.load_state_dict(state_dict=state_dict)
        return rnn


def mdn_loss_fn(y, pi, mu, sigma):
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    loss = torch.exp(m.log_prob(y))
    loss = torch.sum(loss * pi, dim=2)
    loss = -torch.log(loss)
    return loss.mean()


# from SafeWorld
def mdn_loss_stable2(target, pi, mu, sigma):
    '''
    MDN loss with Log-Sum-Exp trick and -inf checks
    '''
    distr = torch.distributions.Normal(loc=mu, scale=sigma)
    log_probs = distr.log_prob(target)
    terms = torch.cat([log_probs, torch.log(pi)], dim=-1).sum(dim=-1)
    max_term, _ = torch.max(terms, dim=-1, keepdim=True)
    logexp = torch.log(torch.exp(terms - max_term).sum(dim=-1))
    result = max_term.squeeze(-1) + logexp
    #  float(inf) seems to be excess and it won't work
    #  result = torch.where((max_term == float('inf')) | (max_term == float('-inf')), max_term.squeeze(-1), result)
    return -torch.mean(result)

def detach(xs):
    return [x.detach() for x in xs]
