import sys
import os
# path hack
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import gym
import argparse
import torch
import torch.nn as nn
import plotly as plt
import plotly.graph_objects as go
from tqdm import tqdm

from worldmodel.agent.Agent import Agent
from worldmodel.VAE.VAE import VAE
from worldmodel.agent.ActorCritic import ControllerAC
from worldmodel.model.MDNRNN import MDNRNN
from params import z_size, n_hidden, n_gaussians, image_height, image_width


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, required=False)
    parser.add_argument('--device', type=str, default='cpu', required=False)
    parser.add_argument('--show', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, required=False)
    parser.add_argument('--restart', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, required=False)
    return parser


def train(epochs, show, restart, action_sz, state_sz, device):
    vae = VAE.load_model('generated/vae.torch', image_channels=3, image_height=image_height, image_width=image_width)
    vae.to(device)
    rnn = MDNRNN.load_model('generated/mdnrnn.torch', z_size, n_hidden, n_gaussians)

    controller = ControllerAC(state_sz, action_sz, n_hidden, device=device)
    if not restart:
        controller = ControllerAC.load_model("generated/actor_critic.torch", state_sz, action_sz, device=device)
    agent = Agent(env, vae, rnn, controller, device=device)

    plot_data = list()
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        reward = agent.rollout(show=show)
        pbar.set_description("Epoch [{}/{}]".format(epoch + 1, epochs))
        pbar.write("Reward: {:.3f}".format(reward))
        plot_data.append(reward)

    controller.save_model("generated/actor_critic.torch")
    plot = go.Figure()
    plot.add_trace(go.Scatter(x=np.arange(epochs), y=np.array(plot_data)))
    plot.show()


if __name__ == "__main__":
    env = gym.make('CarRacing-v0')
    #TODO find better aproach
    action_sz = env.action_space.shape[0]
    state_sz = 32
    args = create_parser().parse_args()

    train(args.epochs, args.show, args.restart, action_sz, state_sz, torch.device(args.device))
