# -*- coding: utf-8 -*-
import os
import gym
import imageio
import numpy as np
from tqdm import tqdm
from random import choice, random, randint
import argparse
import torch

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('--epochs', type=int, default=30, required=False)
    parser.add_argument('--show', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, required=False)
    parser.add_argument('--steps', type=int, default=1000, required=False)
    parser.add_argument('--with-agent', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, required=False)
    return parser 


# TODO decide how to choose action
def get_action(env, steps):
    action = env.action_space.sample()
    #  return action

    if steps < 70:
        action[0] = 0
    action[1] = 1
    action[2] = 0 
    return action


def sample_rollouts(env, epochs, steps, images):
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
            action = get_action(env, t)
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

    sample_rollouts(env, args.epochs, args.steps, images)
    env.close()
