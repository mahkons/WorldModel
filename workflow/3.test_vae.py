import sys
import os
# path hack
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import make_grid

from worldmodel.VAE.VAE import VAE
from params import image_height, image_width

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def compare(model, dataset):
    with torch.no_grad():
        x = dataset[random.randint(1, 100)][0].unsqueeze(0)
        recon_x, _, _ = model(x)
        show(make_grid(torch.cat([x, recon_x])))
        plt.show()

if __name__ == "__main__":
    dataset = datasets.ImageFolder(root='rollouts', transform=transforms.ToTensor())
    vae = VAE.load_model("generated/vae.torch", image_height=image_height, image_width=image_width)
    compare(vae, dataset)
