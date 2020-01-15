import torch
import numpy as np
import random
import math
from collections import namedtuple

from workflow.params import PRIORITY_EPS, PRIORITY_ALPHA, BETA_START, BETA_END, BETA_DECAY

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done', 'model_error'))


class RangeTree:
    def __init__(self, capacity):
        self.size = 1
        while self.size < capacity:
            self.size *= 2
        self.values = np.zeros(2 * self.size)
        self.max_values = np.zeros(2 * self.size)
    
    def add(self, pos, x):
        pos += self.size
        self.values[pos] = x
        self.max_values[pos] = x;
        pos //= 2
        while (pos):
            self.values[pos] = self.values[2 * pos] + self.values[2 * pos + 1]
            self.max_values[pos] = max(self.max_values[2 * pos], self.max_values[2 * pos + 1])
            pos //= 2
            
    def get_max(self):
        return self.max_values[1]

    def get_sum(self):
        return self.values[1]
            
    def get(self, x):
        x *= self.values[1]
        pos = 1
        while pos < self.size:
            if x > self.values[2 * pos]:
                x -= self.values[2 * pos]
                pos = pos * 2 + 1
            else:
                pos = pos * 2
        return pos - self.size


class PrioritizedReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()
        self.position = 0
        self.tree = RangeTree(capacity)
        self.priorities = np.empty((capacity, ))
        self.size = 0
        self.steps_done = 0
    
    def push(self, state, action, next_state, reward, done, model_error):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            
        self.memory[self.position] = Transition(
            state.detach(),
            action.detach(),
            next_state.detach(),
            reward.detach(),
            done.detach(),
            model_error.detach()
        )
        self.priorities[self.position] = max(self.tree.get_max(), PRIORITY_EPS)
        #TODO delete after experiments
        self.priorities[self.position] = (model_error.item() + PRIORITY_EPS) ** PRIORITY_ALPHA
        self.tree.add(self.position, self.priorities[self.position])
        
        self.position += 1
        if self.position == self.capacity:
            self.position = 0
            
    def get_priorities(self, positions):
        sum_priorities = self.tree.get_sum()
        return [self.priorities[pos] / sum_priorities for pos in positions]
    
    def update(self, positions, td_errors):
        for (pos, error) in zip(positions, td_errors):
            self.priorities[pos] = (abs(error) + PRIORITY_EPS) ** PRIORITY_ALPHA
            self.tree.add(pos, self.priorities[pos])

    def sample_positions_uniform(self, batch_size):
        return random.sample(range(len(self.memory)), batch_size)
            
    def sample_positions(self, batch_size):
        positions = [self.tree.get(np.random.uniform(k / batch_size, (k + 1) / batch_size)) for k in range(batch_size)]

        beta = BETA_END - (BETA_END - BETA_START) * math.exp(-1. * self.steps_done / BETA_DECAY)
        self.steps_done += 1

        weights = (torch.tensor(self.get_priorities(positions)) * batch_size) ** (-beta)
        weights /= torch.max(weights) 
        return positions, weights

    def get_transitions(self, positions):
        transitions = [self.memory[pos] for pos in positions]

        batch = Transition(*zip(*transitions))
        state = torch.cat(batch.state)
        action = torch.cat(batch.action)
        reward = torch.cat(batch.reward)
        next_state = torch.cat(batch.next_state)
        done = torch.cat(batch.done)
        model_error = torch.cat(batch.model_error)

        return state, action, reward, next_state, done, model_error

    def __len__(self):
        return len(self.memory)

    def clean(self):
        self.memory = list()
        self.position = 0
        self.tree = RangeTree(self.capacity)
        self.priorities = np.empty((self.capacity, ))
