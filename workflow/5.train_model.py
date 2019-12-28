import sys
import os
# path hack # TODO restructure?
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
from itertools import count
import plotly.graph_objects as go
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from worldmodel.model.MDNRNN import MDNRNN, mdn_loss_fn, detach

device = torch.device("cpu")

epochs = 500
seqlen = 16
BATCH_SIZE = 20

z_size = 32
n_hidden = 256
n_gaussians = 5
plot_data = list()

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, required=False)
    parser.add_argument('--restart', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, required=False)
    parser.add_argument('--device', type=str, default='cpu', required=False)
    parser.add_argument('--learning-rate', type=float, default=1e-3, required=False)
    return parser 


def train(epochs, restart, device, learning_rate):
    model = MDNRNN(z_size, n_hidden, n_gaussians).to(device)
    if not restart:
        model.load_state_dict(torch.load("generated/mdnrnn.torch", map_location='cpu'))
    criterion = mdn_loss_fn
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        pbar.set_description("Epoch [{}/{}]".format(epoch + 1, epochs))

        hidden = model.init_hidden(BATCH_SIZE, device)
        for i in range(0, z.size(1) - seqlen, seqlen):
            inputs = z[:, i:i+seqlen, :]
            targets = z[:, (i+1):(i+1)+seqlen, :]
            
            hidden = detach(hidden)
            (pi, mu, sigma), hidden = model(inputs, hidden)
            loss = criterion(targets.unsqueeze(2), pi, mu, sigma)
            
            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            plot_data.append(loss.item())
            
        pbar.write("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, epochs, loss.item()))

    torch.save(model.state_dict(), "generated/mdnrnn.torch")


if __name__ == "__main__":
    args = create_parser().parse_args()
    device = torch.device(args.device)

    z = torch.load("generated/z.torch")
    mu = torch.load("generated/mu.torch")
    logvar = torch.load("generated/logvar.torch")

    z = z.view(BATCH_SIZE, -1, z.size(2)).to(device)
    train(args.epochs, args.restart, args.device, args.learning_rate)

    plot = go.Figure()
    plot.add_trace(go.Scatter(x=np.arange(len(plot_data)), y=np.array(plot_data)))
    plot.show()
