# -*- coding: utf-8 -*-
import os
import gym
import imageio
import numpy as np
from tqdm import tqdm
from random import choice, random, randint
import argparse

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, required=False)
    parser.add_argument('--show', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, required=False)
    parser.add_argument('--steps', type=int, default=10**10, required=False)
    return parser 


# TODO decide how to choose action
def get_action(env):
    action = env.action_space.sample()
    action[0] /= 10
    action[1] = 1
    action[2] = 0 
    return action


if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    args = create_parser().parse_args()

    os.makedirs('rollouts/SimplePolicy', exist_ok=True) 
    os.system('rm -rf rollouts/SimplePolicy/*')

    pbar = tqdm(range(args.epochs))
    for episode in pbar:
        pbar.set_description('Episode [{}/{}]'.format(episode + 1, args.epochs))

        obs = env.reset()
        for t in range(args.steps):
            action = get_action(env)
            obs, reward, done, _ = env.step(action)
            if args.show:
                env.render()
            if done:
                break

            i = ('0000' + str(t))[-4:]
            imageio.imwrite(f'rollouts/SimplePolicy/car_{episode}_{i}.jpg', obs)
