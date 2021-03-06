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
from worldmodel.agent.ReplayMemory import ReplayMemory
from worldmodel.agent.PrioritizedReplay import PrioritizedReplayMemory
from worldmodel.model.MDNRNN import MDNRNN
from params import z_size, n_hidden, n_gaussians, image_height, image_width, actor_lr, critic_lr, mem_size


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('--epochs', type=int, default=1, required=False)
    parser.add_argument('--device', type=str, default='cpu', required=False)
    parser.add_argument('--show', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, required=False)
    parser.add_argument('--restart', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, required=False)

    parser.add_argument('--no-train', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, required=False)

    parser.add_argument('--memtype', type=str, default='Classic', required=False)
    parser.add_argument('--plot-path', type=str, default='generated/train_plot.torch', required=False)
    return parser


def train(env, epochs, show, restart, action_sz, state_sz, memory, device, plot_path, no_train):
    #  vae = VAE.load_model('generated/vae.torch', image_channels=3, image_height=image_height, image_width=image_width)
    #  vae.to(device)
    vae = None
    rnn = MDNRNN.load_model('generated/mdnrnn.torch', z_size, n_hidden, n_gaussians)
    rnn.to(device)

    controller = ControllerAC(state_sz, action_sz, n_hidden, memory=memory, device=device)
    controller.to(device)
    plot_data = list()

    if not restart:
        controller = ControllerAC.load_model("generated/actor_critic.torch", state_sz, action_sz, n_hidden, memory=memory, device=device, actor_lr=actor_lr, critic_lr=critic_lr)
        #TODO just clean?
        if no_train:
            controller.memory.clean()

        controller.to(device)
        plot_data = torch.load(plot_path)

    agent = Agent(env, vae, rnn, controller, device=device)

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        reward, steps = agent.rollout(show=show, train_agent=not no_train)
        pbar.set_description("Epoch [{}/{}]".format(epoch + 1, epochs))
        pbar.write("Reward: {:.3f}".format(reward))
        plot_data.append((steps, reward))

    controller.save_model("generated/actor_critic.torch")
    torch.save(plot_data, plot_path)
    plot = go.Figure()
    x, y = zip(*plot_data)
    plot.add_trace(go.Scatter(x=np.arange(len(plot_data)), y=np.array(y)))
    plot.show()


def get_memory(memtype):
    if memtype == "Classic":
        return ReplayMemory(mem_size)
    elif memtype == "Prioritized":
        return PrioritizedReplayMemory(mem_size)
    else:
        raise ValueError("unknown memtype")


if __name__ == "__main__":
    args = create_parser().parse_args()
    env = gym.make(args.env) # CarRacing-v0 or LunarLanderContinuous-v2
    action_sz = env.action_space.shape[0]
    state_sz = z_size

    train(env, args.epochs, args.show, args.restart, action_sz, state_sz, get_memory(args.memtype), torch.device(args.device), args.plot_path, args.no_train)
