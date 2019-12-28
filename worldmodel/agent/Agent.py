import numpy as np
import gym
import math
import random
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions.normal import Normal
from torch.nn.utils import clip_grad_norm_

def detach(xs):
    return [x.detach() for x in xs]

class Agent:
    def __init__(self, env, vae, rnn, controller, device, z_size=32):
        self.env = env
        self.vae = vae
        self.rnn = rnn
        self.controller = controller
        self.device = device

        self.action_sz = self.controller.action_sz
        self.z_size = z_size

    def resize_obs(self, obs):
        transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
        ])
        return transform(obs).to(self.device).unsqueeze(0)

    def add_hidden(self, state, hidden):
        return torch.cat([state, hidden[0].squeeze(1)], dim=1)

    def rollout(self, show=False):
        obs = self.resize_obs(self.env.reset())
        hidden = self.rnn.init_hidden(1, self.device)

        state = self.vae.play_encode(obs)
        state = self.add_hidden(state, hidden)

        done = False
        total_reward = 0

        for t in count():
            hidden = detach(hidden)
            if show:
                self.env.render()

            action = self.controller.select_action(state)
            obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            reward = torch.tensor([reward], dtype=torch.float, device=self.device)
            action = torch.tensor([action], dtype=torch.float, device=self.device)

            obs = self.resize_obs(obs)
            next_state = self.vae.play_encode(obs)
            _, next_hidden = self.rnn.play_encode(next_state.unsqueeze(0), hidden)
            next_state = self.add_hidden(next_state, next_hidden)

            self.controller.memory.push(state, action, next_state, reward)
            state, hidden = next_state, next_hidden
            self.controller.optimize()

            if done:
                break;
        return total_reward
