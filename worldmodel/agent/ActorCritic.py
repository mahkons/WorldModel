import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

from worldmodel.agent.ReplayMemory import ReplayMemory, Transition

HIDDEN_ACTOR_SIZE = 256
HIDDEN_CRITIC_SIZE = 256
GAMMA = 0.999
BATCH_SIZE = 64
TAU = 0.001

class Actor(nn.Module):
    def __init__(self, state_sz, action_sz):
        super(Actor, self).__init__()              
        self.fc1 = nn.Linear(state_sz, HIDDEN_ACTOR_SIZE)
        self.fc2 = nn.Linear(HIDDEN_ACTOR_SIZE, action_sz)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.tanh(x)


class Critic(nn.Module):
    def __init__(self, state_sz, action_sz):
        super(Critic, self).__init__()              
        self.fc1 = nn.Linear(state_sz + action_sz, HIDDEN_CRITIC_SIZE)
        self.fc2 = nn.Linear(HIDDEN_CRITIC_SIZE, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = self.fc2(x)
        return x


class ControllerAC(nn.Module):
    def __init__(self, state_sz, action_sz, mem_size=1000000, actor_lr=1e-4, critic_lr=1e-4, device='cpu'):
        super(ControllerAC, self).__init__()
        self.state_sz = state_sz
        self.action_sz = action_sz
        self.memory = ReplayMemory(mem_size)
        self.device = device

        self.actor = Actor(state_sz, action_sz).to(device)
        self.target_actor = Actor(state_sz, action_sz).to(device)

        self.critic = Critic(state_sz, action_sz).to(device)
        self.target_critic = Critic(state_sz, action_sz).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.steps_done = 0

    def select_action(self, state):
        with torch.no_grad():
            return self.actor(state).to(torch.device('cpu')).numpy().squeeze(0)

    def hard_update(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def soft_update_net(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

    def soft_update(self):
        self.soft_update_net(self.actor, self.target_actor)
        self.soft_update_net(self.critic, self.target_critic)

    def optimize_critic(self):
        if len(self.memory) < BATCH_SIZE:
            return
        state, action, reward, next_state = self.memory.sample(BATCH_SIZE)

        state_action_values = self.critic(state, action)
        with torch.no_grad():
            noise = torch.empty(action.shape).data.normal_(0, 0.2).to(self.device)
            noise = noise.clamp(-0.5, 0.5)

            next_action = (self.target_actor(next_state) + noise).clamp(-1., 1.)
            next_values = self.target_critic(next_state, next_action).squeeze(1)
        
        expected_state_action_values = (next_values * GAMMA) + reward

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def optimize_actor(self):
        if len(self.memory) < BATCH_SIZE:
            return
        state, action, reward, next_state = self.memory.sample(BATCH_SIZE)
        predicted_action = self.actor(state)
        
        loss = -torch.sum(self.critic(state, predicted_action), dim=1).mean()
        
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def optimize(self):
        self.optimize_critic()
        self.optimize_actor()
        self.soft_update()

    def save_model(self, path):
        torch.save(self, path)

    @staticmethod
    def load_model(path, *args, **kwargs):
        cnt = torch.load(path, map_location='cou')
        return cnt
