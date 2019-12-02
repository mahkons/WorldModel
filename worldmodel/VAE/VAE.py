import typing

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, input):
        return input.view(-1, self.size, 1, 1)


class VAE(nn.Module):
    # PyCharm problems
    # https://youtrack.jetbrains.com/issue/PY-37601
    def __call__(self, *inp, **kwargs) -> typing.Any:
        return super().__call__(*inp, **kwargs)

    def __init__(
            self,
            image_height: int = 64,
            image_width: int = 64,
            image_channels: int = 3,
            h_dim: int = 1024,
            z_dim: int = 32,
            device: str = "cpu"
    ):
        super(VAE, self).__init__()
        self.device = device
        self.h = image_height
        self.w = image_width
        self.c = image_channels
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(h_dim),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

        self.to(self.device)

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def reparameterize(self, mu: torch.Tensor, logstd: torch.Tensor):
        if self.training:
            std = (logstd * 0.5).exp_()
            std_prob = torch.randn(*mu.size(), device=self.device)
            return mu + std_prob * std
        else:
            return mu   # inference time

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        return self.decoder(z), mu, logvar

    @staticmethod
    def calculate_loss(pred_x: torch.Tensor, true_x: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor):
        bce = F.mse_loss(pred_x, true_x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logstd - mu.pow(2) - logstd.exp())
        return bce + kld

    def play_encode(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            obs = to_tensor(obs).unsqueeze(0).float()
            return self.reparameterize(*self.encode(obs)).squeeze()

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load_model(cls, path, *args, **kwargs):
        state_dict = torch.load(path)
        vae = cls(*args, **kwargs)
        vae.load_state_dict(state_dict=state_dict)
        return vae
