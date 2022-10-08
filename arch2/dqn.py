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

    # state = Variable(torch.FloatTensor(np.float32(state)))
    # next_state = Variable(torch.FloatTensor(np.float32(next_state)).squeeze(1), requires_grad=True)
    # action = Variable(torch.LongTensor(action))
    # reward = Variable(torch.FloatTensor(reward))
    # done = Variable(torch.FloatTensor(done))
    
    # implement the loss function here

    # Make predictions
    indices = np.arange(batch_size)
    state_q_values = model(state)
    next_states_q_values = model(next_state)
    next_states_target_q_values = target_model(next_state)

    # print(state_q_values,action.unsqueeze(1))
    # Find selected action's q_value
    selected_q_value = model.forward(state)[indices, action]
    q_next = target_model.forward(next_state).max(dim=1)[0]
    #q_next[int(done)]= 0.0

    #selected_q_value = state_q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    #selected_q_value = state_q_values.index_select(1, action)
    # Get indice of the max value of next_states_q_values
    # Use that indice to get a q_value from next_states_target_q_values
    # We use greedy for policy So it called off-policy
    
    #next_states_target_q_value = next_states_target_q_values.gather(1, next_states_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
    
    #next_states_target_q_value = next_states_target_q_values.max(dim=1)[0]
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
        #state, action, reward, next_state, done = list(self.buffer)[(np.random.randint(0, len(self.buffer)-1))]
        #transitions = random.sample(self.buffer, batch_size)
        # print(np.array(transitions).shape)
        # Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state','done'])
        #batch = list(zip(*transitions))
        # print(batch)
        #state, action, reward,  next_state, done = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.array(batch[3]), np.array(batch[4])
        #return state, action, reward, next_state, done
        return zip(*random.sample(self.buffer, batch_size))
    def __len__(self):
        return len(self.buffer)
