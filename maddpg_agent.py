import numpy as np
from ddpg_agent import Agent
from buffer import ReplayBuffer
from collections import deque,namedtuple
import torch
import torch.nn.functional as F
import random

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

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
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, 2) #defined in the function setup
        self.t_step = 0
    
    def act(self, states, noise=0.0):
        actions = [agent.act(state) for agent, state in zip(self.maddpg, states)]
        return actions
    
    def target_act(self, states, noise=0.0):
        target_actions = [agent.target_act(state) for agent, state in zip(self.maddpg, states)]
        return target_actions
        
    def step(self, state, action, reward, next_state, done, Batch = 256):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % Update_every
        if len(self.memory) > Batch and self.t_step == 0:
            for _ in range(Update_times):
                experiences = [self.memory.sample() for _ in range(2)]
                [self.learn(experiences[agent], agent, gamma = 0.99) for agent in range(2)]
    def learn(self, experiences, agent_number, gamma):
        states, actions, rewards, next_states, dones = experiences
        states_1 = states[agent_number:len(states):2]
        next_states_1 = next_states[agent_number:len(states):2]
        rewards_1 = rewards[:,agent_number]
        dones_1 = dones[:,agent_number]
        actions_1 = actions[agent_number:len(states):2].view(256,2)
        agent = self.maddpg[agent_number]
        agent.critic_optimizer.zero_grad()
        actions_next = agent.actor_target(next_states_1.to(device))
        critic_input = torch.cat((next_states_1, actions_next.float()), dim = 1).to(device)
        Q_targets_next = agent.critic_target(critic_input).detach().view(256)
        Q_targets = rewards_1 + (gamma*torch.dot(Q_targets_next,(1-dones_1)))
        critic_input = torch.cat((states_1, actions_1), dim = 1).to(device)
        Q_expected = agent.critic_local(critic_input).view(256)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()
        actions_pred = (agent.act(states_1.to('cpu').numpy())).squeeze(0)
        actions_pred = torch.from_numpy(actions_pred).to(device)
        
        critic_input = torch.cat((states_1.float(), (actions_pred).float()), dim = 1).to(device)
        actor_loss = -agent.critic_local(critic_input).mean()
        
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()
        
        self.soft_update(agent.critic_local, agent.critic_target)
        self.soft_update(agent.actor_local, agent.actor_target)
    def soft_update(self, local_model, target_model, tau=1e-3):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
            


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    
    