import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

from worldmodel.agent.ReplayMemory import Transition
from workflow.params import GAMMA, TAU, BATCH_SIZE, PRIORITY_DECR, MIN_MEMORY


class Actor(nn.Module):
    def __init__(self, state_sz, action_sz, hidden_sz):
        super(Actor, self).__init__()              
        self.fc1 = nn.Linear(state_sz + hidden_sz, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_sz)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.tanh(x)


class Critic(nn.Module):
    def __init__(self, state_sz, action_sz, hidden_sz):
        super(Critic, self).__init__()              
        self.fc1 = nn.Linear(state_sz + hidden_sz + action_sz, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ControllerAC(nn.Module):
    def __init__(self, state_sz, action_sz, hidden_sz, memory, actor_lr=1e-4, critic_lr=1e-4, device='cpu'):
        super(ControllerAC, self).__init__()
        self.state_sz = state_sz
        self.action_sz = action_sz
        self.hidden_sz = hidden_sz
        self.memory = memory
        self.device = device

        self.actor = Actor(state_sz, action_sz, hidden_sz).to(device)
        self.target_actor = Actor(state_sz, action_sz, hidden_sz).to(device)

        self.critic = Critic(state_sz, action_sz, hidden_sz).to(device)
        self.target_critic = Critic(state_sz, action_sz, hidden_sz).to(device)

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

    def combine_errors(self, td_error, model_error):
        #  return td_error
        return model_error

    def optimize_critic(self, positions, weights):
        state, action, reward, next_state, done, model_error = self.memory.get_transitions(positions)

        state_action_values = self.critic(state, action)
        with torch.no_grad():
            noise = torch.empty(action.shape).data.normal_(0, 0.2).to(self.device) # from SafeWorld
            noise = noise.clamp(-0.5, 0.5)

            next_action = (self.target_actor(next_state) + noise).clamp(-1., 1.)
            next_values = self.target_critic(next_state, next_action).squeeze(1)
        
            expected_state_action_values = (next_values * GAMMA * (1 - done)) + reward
            td_error = (expected_state_action_values.unsqueeze(1) - state_action_values).squeeze(1) # TODO clamp add?
            self.memory.update(positions, self.combine_errors(torch.abs(td_error), torch.abs(model_error)))

        loss = F.smooth_l1_loss(state_action_values.squeeze() * weights, expected_state_action_values * weights)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def optimize_actor(self, positions, weights):
        state, action, reward, next_state, done, model_error = self.memory.get_transitions(positions)
        predicted_action = self.actor(state)

        value = self.critic(state, predicted_action) # TODO remove or not remove (1 - done)?
        loss = -(value.squeeze() * weights).mean()
        
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def optimize(self):
        if len(self.memory) < MIN_MEMORY:
            return
        positions, weights = self.memory.sample_positions(BATCH_SIZE)
        weights = weights.to(self.device)

        self.optimize_critic(positions, weights)
        self.optimize_actor(positions, weights)
        self.soft_update()

    def save_model(self, path):
        torch.save(self, path)

    @staticmethod
    def load_model(path, *args, **kwargs):
        cnt = torch.load(path, map_location='cpu')
        cnt.to(cnt.device)
        cnt.actor_optimizer = torch.optim.Adam(cnt.actor.parameters(), lr=kwargs['actor_lr'])
        cnt.critic_optimizer = torch.optim.Adam(cnt.critic.parameters(), lr=kwargs['critic_lr'])
        return cnt
