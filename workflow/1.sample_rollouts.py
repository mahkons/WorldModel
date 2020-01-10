# -*- coding: utf-8 -*-
import os
import gym
import imageio
import numpy as np
from tqdm import tqdm
from random import choice, random, randint
import argparse
import torch

from params import z_size, n_hidden, n_gaussians, image_height, image_width, actor_lr, critic_lr, mem_size

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('--epochs', type=int, default=30, required=False)
    parser.add_argument('--show', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, required=False)
    parser.add_argument('--steps', type=int, default=1000, required=False)
    parser.add_argument('--with-agent', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, required=False)
    return parser 


# TODO decide how to choose action
def get_action_randomly(env, steps, obs):
    action = env.action_space.sample()
    #  return action

    if steps < 70:
        action[0] = 0
    action[1] = 1
    action[2] = 0 
    return action


class GetActionWithAgent:
    def __init__(self):
        device = torch.device('cpu')
        self.controller = ControllerAC.load_model(
                "generated/actor_critic.torch", 
                state_sz,
                action_sz,
                n_hidden,
                memory=memory,
                device=device,
                actor_lr=actor_lr,
                critic_lr=critic_lr
            )

        self.vae = VAE.load_model('generated/vae.torch', image_channels=3, image_height=image_height, image_width=image_width)

    def __call__(self, env, steps, obs):
        state = self.vae.play_encode(obs)
        action = self.controller.select_action(state)


def sample_rollouts(env, epochs, steps, images, get_action):
    if images:
        os.makedirs('rollouts/SimplePolicy', exist_ok=True) 
        os.system('rm -rf rollouts/SimplePolicy/*')
    else:
        os.makedirs("generated", exist_ok=True)

    z_l = list()
    pbar = tqdm(range(epochs))
    for episode in pbar:
        pbar.set_description('Episode [{}/{}]'.format(episode + 1, epochs))

        obs = env.reset()
        for t in range(steps):
            action = get_action(env, t, obs)
            obs, reward, done, _ = env.step(action)
            if args.show:
                env.render()
            if done:
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

    if env_name in ['CarRacing-v0']:
        images = True
    elif env_name in ['LunarLanderContinuous-v2']:
        images = False
    else:
        raise ValueError('Unknown env')

    if args.with_agent:
        get_action = GetActionWithAgent
    else:
        get_action = get_action_randomly

    sample_rollouts(env, args.epochs, args.steps, images, get_action)
    env.close()
