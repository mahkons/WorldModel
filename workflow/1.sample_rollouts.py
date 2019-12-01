# -*- coding: utf-8 -*-
import gym
import imageio
import numpy as np
from tqdm import tqdm
from random import choice, random, randint
import argparse

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1, required=False)
    parser.add_argument('--show', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, required=False)
    parser.add_argument('--steps', type=int, default=10**10, required=False)
    return parser 


def get_action(env):
    return env.action_space.sample()

if __name__ == "__main__":
    env = gym.make("CarRacing-v0")
    args = create_parser().parse_args()

    pbar = tqdm(range(args.episodes))
    for episode in pbar:
        pbar.set_description("Episode [{}/{}]".format(episode + 1, args.episodes))

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
