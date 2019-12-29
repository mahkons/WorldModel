import torch
import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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
        self.tree = RangeTree()
        self.priorities = np.empty((capacity, ))
        self.size = 0
    
    def push(self, state, action, next_state, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            
        self.memory[self.position] = Transition(
            state.clone().detach(),
            action.clone().detach(),
            next_state.clone().detach(),
            reward.clone().detach()
        )
        self.priorities[self.position] = max(self.tree.get_max(), PRIORITY_EPS)
        self.tree.add(self.position, self.priorities[self.position] ** PRIORITY_ALPHA)
        
        self.position += 1
        if self.position == self.capacity:
            self.position = 0
            
    def get_priorities(self, positions):
        return [self.priorities[pos] for pos in positions]
    
    def get_transitions(self, positions):
        return [self.memory[pos] for pos in positions]
            
    def update(self, positions, td_errors):
        for (pos, error) in zip(positions, td_errors):
            self.priorities[pos] = abs(error) + PRIORITY_EPS
            self.tree.add(pos,  self.priorities[pos] ** PRIORITY_ALPHA)
            
    def sample(self, batch_size):
        return [self.tree.get(np.random.uniform(k / batch_size, (k + 1) / batch_size)) for k in range(batch_size)]

    def __len__(self):
        return len(self.memory)
