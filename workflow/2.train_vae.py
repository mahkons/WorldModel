import sys
import os
# path hack # TODO restructure?
sys.path.insert(0, os.path.abspath('..'))

import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
import plotly.graph_objects as go
import numpy as np

from worldmodels.VAE.VAE import VAE

BATCH_SIZE = 32
plot_data = list()

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, required=False)
    parser.add_argument('--restart', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, required=False)
    parser.add_argument('--device', type=str, default='cpu', required=False)
    parser.add_argument('--learning-rate', type=float, default=1e-3, required=False)
    return parser 


def train(epochs, restart, device, dataloader, learning_rate):
    model = VAE(image_height=64, image_width=64, h_dim=1024, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if not restart:
        model.load_state_dict(torch.load("generated/vae.torch", map_location='cpu'))
    model.to(device)

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        pbar.set_description("Epoch [{}/{}]".format(epoch + 1, epochs))

        for idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            recon_images, mu, logstd = model(images)

            loss = VAE.calculate_loss(recon_images, images, mu, logstd)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            plot_data.append(loss.item() / BATCH_SIZE)
            pbar.write("Loss: {:.3f}".format(loss.item() / BATCH_SIZE))

    model.save_model('generated/vae.torch')


if __name__ == "__main__":
    #  dataset = datasets.ImageFolder(root='rollouts', transform=transforms.ToTensor())
    # TODO work not only with images 64 * 64
    dataset = datasets.ImageFolder(root='rollouts', transform=transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(), 
    ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    args = create_parser().parse_args()
    train(args.epochs, args.restart, torch.device(args.device), dataloader, args.learning_rate)

    plot = go.Figure()
    plot.add_trace(go.Scatter(x=np.arange(len(plot_data)), y=np.array(plot_data)))
    plot.show() 
