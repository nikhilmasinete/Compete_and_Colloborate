import numpy as np
from ddpg_agent import Agent
from collections import deque,namedtuple
import torch
import torch.nn.functional as F
import random

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256      # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3       # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
num_agents = 2
Update_every = 1
Update_times = 1

epsilon = 1.0
epsilon_decay = 1e-6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG():
    def __init__(self, state_size, action_size, discount_factor=0.95, tau=0.02):
       
        self.num_agents = 2
        self.maddpg = [Agent(state_size,action_size,2) for i in range(self.num_agents)]
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, num_agents) #defined in the function setup
        self.t_step = 0
    
    def act(self, states, noise=0.0):
        actions = [agent.act(state, add_noise=True) for agent, state in zip(self.maddpg, states)]
        return actions
    
    def target_act(self, states, noise=0.0):
        actions = [agent.actor_target(state) for agent, state in zip(self.maddpg, states)]
        return actions
        
    def step(self, state, action, reward, next_state, done, Batch = 256):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % Update_every
        if len(self.memory) > Batch and self.t_step == 0:
            experiences = self.memory.sample()
            for _ in range(Update_times):
                
                [self.learn(experiences, agent, gamma = 0.99) for agent in range(2)]
    def learn(self, experiences, agent_number, gamma):
        states, actions, rewards, next_states, dones = experiences
        agent = self.maddpg[agent_number]
        all_states = torch.cat(states, dim=1).to(device)
        all_next_states = torch.cat(next_states, dim=1).to(device)
        all_actions = torch.cat(actions, dim=1).to(device)
        
        next_actions = [actions[index].clone() for index in range(2)]
        next_actions[agent_number] = agent.actor_target(next_states[agent_number])
        all_next_actions = torch.cat(next_actions, dim=1).to(device)
        
        Q_target_next = agent.critic_target(all_next_states,all_next_actions)
        Q_target = rewards[agent_number] + GAMMA * Q_target_next *(1-dones[agent_number])
        Q_expected = agent.critic_local(all_states,all_actions)
        critic_loss = F.mse_loss(Q_expected,Q_target)
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()
        
        actions_pred = [actions[index].clone() for index in range(2)]
        actions_pred[agent_number] = agent.actor_local(states[agent_number])
        all_actions_pred = torch.cat(actions_pred, dim=1).to(device)
        
        agent.actor_optimizer.zero_grad()
        actor_loss = -agent.critic_local(all_states, all_actions_pred).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor_local.parameters(), 1)
        agent.actor_optimizer.step()
        
        self.soft_update(agent.critic_local, agent.critic_target)
        self.soft_update(agent.actor_local, agent.actor_target)
        
    def soft_update(self, local_model, target_model, tau=1e-2):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
            


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, num_agents, seed=2):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.seed = random.seed(seed)
        self.num_agents = num_agents
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)


        states = [torch.from_numpy(np.vstack([e.states[index] for e in experiences if e is not None])).float().to(device) for index in range(num_agents)]
        actions = [torch.from_numpy(np.vstack([e.actions[index] for e in experiences if e is not None])).float().to(device) for index in range(num_agents)]
        rewards = [torch.from_numpy(np.vstack([e.rewards[index] for e in experiences if e is not None])).float().to(device) for index in range(num_agents)]
        next_states = [torch.from_numpy(np.vstack([e.next_states[index] for e in experiences if e is not None])).float().to(device) for index in range(num_agents)]
        dones = [torch.from_numpy(np.vstack([e.dones[index] for e in experiences if e is not None]).astype(np.uint8)).float().to(device) for index in range(num_agents)]
        
        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
