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

    #TODO
    # code for carRacing. think of better way
    #  def transform_obs(self, obs):
        #  transform = T.Compose([
            #  T.ToPILImage(),
            #  T.ToTensor(),
        #  ])
        #  obs = transform(obs).to(self.device).unsqueeze(0)
        #  return self.vae.play_encode(obs)

    def transform_obs(self, obs):
        return torch.tensor([obs], device=self.device)

    def add_hidden(self, state, hidden):
        return torch.cat([state, hidden[0].squeeze(1)], dim=1)

    def wrap(self, x):
        return torch.tensor([x], dtype=torch.float, device=self.device)

    def rollout(self, show=False):
        state = self.transform_obs(self.env.reset())
        hidden = self.rnn.init_hidden(1, self.device)
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
            reward = self.wrap(reward)
            action = self.wrap(action)

            next_state = self.transform_obs(obs)
            _, next_hidden = self.rnn.play_encode(next_state.unsqueeze(0), hidden)
            next_state = self.add_hidden(next_state, next_hidden)

            self.controller.memory.push(state, action, next_state, reward, self.wrap(done))
            state, hidden = next_state, next_hidden
            self.controller.optimize()

            if done:
                break;
        return total_reward
