import sys
import os
# path hack # TODO restructure?
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision.utils import make_grid

from worldmodel.model.MDNRNN import MDNRNN, mdn_loss_fn, detach
from worldmodel.VAE.VAE import VAE
from params import z_size, n_hidden, n_gaussians, image_height, image_width

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def compare(x):
    with torch.no_grad():
        show(make_grid(x))
        plt.show()

BATCH_SIZE = 20

if __name__ == "__main__":
    device = torch.device("cpu")

    z = torch.load("generated/z.torch")
    z = z.view(BATCH_SIZE, -1, z.size(2)).to(device)

    vae_model = VAE(image_width=image_width, image_height=image_height, image_channels=3).to(device)
    vae_model.load_state_dict(torch.load('generated/vae.torch', map_location='cpu'))

    model = MDNRNN(z_size, n_hidden, n_gaussians)
    model.load_state_dict(torch.load('generated/mdnrnn.torch', map_location='cpu'))

    zero = np.random.randint(z.size(0))
    one = np.random.randint(z.size(1))
    x = z[zero:zero+1, one:one+1, :]
    y = z[zero:zero+1, one+1:one+2, :]

    hidden = model.init_hidden(1, device)
    (pi, mu, sigma), _ = model(x, hidden)

    y_preds = [torch.normal(mu, sigma)[:, :, i, :] for i in range(n_gaussians)]

    compare(vae_model.decode(torch.cat([x, y] + y_preds)))
