import numpy as np
import random
import copy
from collections import namedtuple, deque

from network import Network
from buffer import ReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim

Buffer = int(1e6)
Batch = 128
Gamma = 0.95
Tau = 1e-3
LR_Actor = 1e-4
LR_Critic = 1e-3
Weight_Decay = 0.99

Update_every = 20
Update_times = 10

epsilon = 1.0
epsilon_decay = 1e-6

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, hidden_in_actor, hidden_out_actor, action_size, in_critic, hidden_in_critic, hidden_out_critic, random_seed = 2):
        super(Agent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.actor_local = Network(state_size, hidden_in_actor, hidden_out_actor, action_size, actor=True).to(device)
        self.critic_local = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        self.actor_target = Network(state_size, hidden_in_actor, hidden_out_actor, action_size, actor=True).to(device)
        self.critic_target = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        
        self.soft_update(self.actor_local, self.actor_target, 1)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = LR_Actor)
        
        self.soft_update(self.critic_local, self.critic_target, 1)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = LR_Critic)
        
        self.noise = OUNoise(action_size, random_seed)
        self.t_step = 0
        self.epsilon = epsilon
        
        self.memory = ReplayBuffer(action_size, Buffer, Batch, random_seed)
        
    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        self.epsilon = self.epsilon - epsilon_decay
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action = action + np.maximum(self.epsilon, 0.2)*self.noise.sample()
        return np.clip(action, -1, 1)
    
    def target_act(self, state, add_noise=True):
        #state = torch.from_numpy(state).float().to(device)
        self.actor_target.eval()
        self.epsilon = self.epsilon - epsilon_decay
        with torch.no_grad():
            action = self.actor_target(state).cpu().data.numpy()
        self.actor_target.train()
        if add_noise:
            action = action + np.maximum(self.epsilon, 0.2)*self.noise.sample()
        return np.clip(action, -1, 1)
    def reset(self):
        self.noise.reset()
        
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
        

class OUNoise:
    def __init__(self, size, seed, mu=0., theta = 0.15, sigma = 0.2):
        self.mu = mu*np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
    def reset(self):
        self.state = copy.copy(self.mu)
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * (np.random.rand(*x.shape)-0.5)
        self.state = x + dx
        return self.state
    
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return(states, actions, rewards, next_states, dones)
    def __len__(self):
        return len(self.memory)