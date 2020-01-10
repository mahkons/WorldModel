import sys
import os
# path hack # TODO restructure?
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import make_grid

from worldmodel.VAE.VAE import VAE
from params import image_height, image_width

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

BATCH_SIZE = 1

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', required=False)
    return parser 


def encode():
    z_l = list()
    for (images, _) in tqdm(dataloader):
        z = vae.play_encode(images)
        z_l.append(z)

    z_l = torch.stack(z_l)
    torch.save(z_l, "generated/z.torch")

    with torch.no_grad():
        x = np.random.randint(z.size(0))
        show(make_grid(vae.decode(z_l[x:x+16])))
        plt.show()


if __name__ == "__main__":
    args = create_parser().parse_args()

    dataset = datasets.ImageFolder(root='rollouts', transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    len(dataset.imgs), len(dataloader)

    vae = VAE(image_height=image_height, image_width=image_width, device=torch.device(args.device))
    vae.load_state_dict(torch.load('generated/vae.torch', map_location='cpu'))

    encode()
