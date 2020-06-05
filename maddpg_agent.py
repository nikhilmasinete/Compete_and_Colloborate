import numpy as np
from ddpg_agent import Agent
from buffer import ReplayBuffer
from collections import deque,namedtuple
import torch
import torch.nn.functional as F
import random

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256      # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 3e-4         # learning rate of the actor 
LR_CRITIC = 3e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
num_agents = 2
Update_every = 2
Update_times = 1

epsilon = 1.0
epsilon_decay = 1e-6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG():
    def __init__(self, state_size, action_size, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()
        self.maddpg = [Agent(state_size,action_size,2),
                      Agent(state_size,action_size,2)]
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, num_agents, 2) #defined in the function setup
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
            for _ in range(Update_times):
                experiences = self.memory.sample()
                [self.learn(experiences, agent, gamma = 0.99) for agent in range(2)]
    def learn(self, experiences, agent_number, gamma):
        states, actions, rewards, next_states, dones = experiences
        actions = actions.view(2*BATCH_SIZE,2)
        states_1 = states[agent_number:len(states):2]
        next_states_1 = next_states[agent_number:len(states):2]
        rewards_1 = rewards
        dones_1 = dones
        all_states = torch.cat((states[0:len(states):2], states[1:len(states):2]), dim = 1)
        all_actions = torch.cat((actions[0:len(actions):2], actions[1:len(actions):2]), dim = 1)
        all_next_states = torch.cat((next_states[0:len(next_states):2], next_states[1:len(next_states):2]), dim = 1)
        
#        actions_1 = actions[agent_number:len(states):2].view(256,2)
        agent = self.maddpg[agent_number]
        agent.critic_optimizer.zero_grad()
        
        all_next_actions = torch.cat((self.maddpg[agent_number].actor_target(next_states[0:len(actions):2]), self.maddpg[agent_number].actor_target(next_states[1:len(actions):2])), dim = 1)
        # Critic training
        critic_input = torch.cat((all_next_states, all_next_actions), dim = 1).to(device)

        Q_targets_next = agent.critic_target(all_states, all_actions).detach()
        Q_targets = rewards_1[:,agent_number].view(BATCH_SIZE,1) + gamma*Q_targets_next*(1-dones_1[:,agent_number].view(BATCH_SIZE,1))

        Q_expected = agent.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()
        
        # Actor training
        
        states_1 = states_1.cpu().data.numpy()
        all_actions_pred =  torch.cat((self.maddpg[agent_number].actor_local(states[0:len(actions):2]), self.maddpg[agent_number].actor_local(states[1:len(actions):2])), dim = 1)
        
        

        actor_loss = -agent.critic_local(all_states, all_actions_pred).mean()
        
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()
        
        self.soft_update(agent.critic_local, agent.critic_target)
        self.soft_update(agent.actor_local, agent.actor_target)
        
    def soft_update(self, local_model, target_model, tau=8e-2):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
            


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, num_agents, seed):
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


        states = torch.from_numpy(np.vstack([e.states for e in experiences if e is not None])).float().to(device)
        states_t = torch.from_numpy(np.vstack([e.states for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.actions for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_states for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)