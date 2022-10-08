from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math, random
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.uniform(0,1) <= epsilon:
            action = random.randrange(self.env.action_space.n)
        else:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0),requires_grad=True)
            # TODO: Given state, you should write code to get the Q value and chosen action
            q_values = self.forward(state) 
            action = torch.argmax(q_values).item()
            
        return action

    def copy_from(self, target):
        self.load_state_dict(target.state_dict())

        
def compute_td_loss(model, target_model, batch_size, gamma, replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    
    state = np.concatenate(state)
    next_state = np.concatenate(next_state)
    
    # print(state.shape, next_state.shape)

    state = torch.tensor(state, dtype=torch.float, device=DEVICE)
    next_state = torch.tensor(next_state, dtype=torch.float, device=DEVICE)
    action = torch.tensor(action, dtype=torch.long, device=DEVICE)
    reward = torch.tensor(reward, dtype=torch.float, device=DEVICE)
    done = torch.tensor(done, dtype=torch.float, device=DEVICE)

    # Make predictions
    indices = np.arange(batch_size)
    state_q_values = model(state)
    next_states_q_values = model(next_state)
    next_states_target_q_values = target_model(next_state)

    # print(state_q_values,action.unsqueeze(1))
    # Find selected action's q_value
    selected_q_value = model.forward(state)[indices, action]
    q_next = target_model.forward(next_state).max(dim=1)[0]
   
    # Use Bellman function to find expected q value
    expected_q_value = reward + gamma * q_next * (1 - done)

    # Calc loss with expected_q_value and q_value
    loss = (selected_q_value - expected_q_value.detach()).pow(2).mean()

    
    return loss


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        #state = np.expand_dims(state, 0)
        #next_state = np.expand_dims(next_state, 0)

        self.buffer.append([state[None,: ], action, reward, next_state[None, :], done])

    def sample(self, batch_size):
        # TODO: Randomly sampling data with specific batch size from the buffer
        return zip(*random.sample(self.buffer, batch_size))
    def __len__(self):
        return len(self.buffer)
