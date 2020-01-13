# -*- coding: utf-8 -*-
import sys
import os
# path hack
sys.path.insert(0, os.path.abspath('..'))

import gym
import imageio
import numpy as np
from tqdm import tqdm
from random import choice, random, randint
import argparse
import torch

from params import z_size, n_hidden, n_gaussians, image_height, image_width, actor_lr, critic_lr, mem_size
from worldmodel.agent.ActorCritic import ControllerAC
from worldmodel.model.MDNRNN import MDNRNN

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--iters', type=int, default=10000, required=False) 
    parser.add_argument('--show', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, required=False)
    parser.add_argument('--steps', type=int, default=1000, required=False)
    parser.add_argument('--with-agent', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, required=False)
    return parser 


# TODO decide how to choose action
def get_action_randomly(env, steps, obs):
    action = env.action_space.sample()
    return action

    if steps < 70:
        action[0] = 0
    action[1] = 1
    action[2] = 0 
    return action


class GetActionWithAgent:
    def __init__(self, env, device):
        self.device = device
        action_sz = env.action_space.shape[0]

        self.controller = ControllerAC.load_model(
                "generated/actor_critic.torch", 
                z_size,
                action_sz,
                n_hidden,
                memory=None,
                device=self.device,
                actor_lr=actor_lr,
                critic_lr=critic_lr
            )
        self.controller.to(self.device)

        self.rnn = MDNRNN.load_model('generated/mdnrnn.torch', z_size, n_hidden, n_gaussians)
        self.rnn.to(self.device)
        self.hidden = self.rnn.init_hidden(1, self.device)

    def init_hidden(self):
        self.hidden = self.rnn.init_hidden(1, self.device)

    def add_hidden(self, state, hidden):
        return torch.cat([state, hidden[0].squeeze(1)], dim=1)

    def __call__(self, env, steps, obs):
        state = torch.tensor([obs], device=self.device)
        _, self.hidden = self.rnn.play_encode(state.unsqueeze(0), self.hidden)
        state = self.add_hidden(state, self.hidden)
        action = self.controller.select_action(state)
        return action


def sample_rollouts(env, iters, steps, images, with_agent, get_action):

    if images:
        os.makedirs('rollouts/SimplePolicy', exist_ok=True) 
        os.system('rm -rf rollouts/SimplePolicy/*')
    else:
        os.makedirs("generated", exist_ok=True)

    z_l = list()
    pbar = tqdm(range(iters))
    cnt_iter = 0
    for episode in pbar:
        pbar.set_description('Episode [{}/{}]'.format(cnt_iter + 1, iters))

        if with_agent:
            get_action.init_hidden()

        obs = env.reset()
        for t in range(steps):
            action = get_action(env, t, obs)
            obs, reward, done, _ = env.step(action)
            if args.show:
                env.render()
            if done:
                cnt_iter += t
                pbar.update(t)
                break

            if images:
                i = ('0000' + str(t))[-4:]
                imageio.imwrite(f'rollouts/SimplePolicy/car_{episode}_{i}.jpg', obs)
            else:
                z_l.append(torch.tensor([obs]))

    if not images:
        z_l = torch.stack(z_l)
        torch.save(z_l, "generated/z.torch")



if __name__ == '__main__':
    args = create_parser().parse_args()
    env_name = args.env
    env = gym.make(env_name)

    device = torch.device(args.device)

    if env_name in ['CarRacing-v0']:
        images = True
    elif env_name in ['LunarLanderContinuous-v2']:
        images = False
    else:
        raise ValueError('Unknown env')

    if args.with_agent:
        get_action = GetActionWithAgent(env, device)
    else:
        get_action = get_action_randomly

    sample_rollouts(env, args.iters, args.steps, images, args.with_agent, get_action)
    env.close()
