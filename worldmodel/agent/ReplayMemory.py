import torch
import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()
        self.position = 0
    
    def push(self, state, action, next_state, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(
            state.clone().detach(),
            action.clone().detach(),
            next_state.clone().detach(),
            reward.clone().detach()
        )
        
        self.position += 1
        if self.position == self.capacity:
            self.position = 0
            
    def sample_transitions(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sample(self, batch_size):
        transitions = self.sample_transitions(batch_size)

        batch = Transition(*zip(*transitions))
        state = torch.cat(batch.state)
        action = torch.cat(batch.action)
        reward = torch.cat(batch.reward)
        next_state = torch.cat(batch.next_state)

        return state, action, reward, next_state

    def __len__(self):
        return len(self.memory)

