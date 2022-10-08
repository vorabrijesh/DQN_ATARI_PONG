from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import math, random
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class QLearner(nn.Module):
    def __init__(self,env, batch_size, gamma,replay_buffer):
        super(QLearner, self).__init__()

        # self.checkpoint_dir = chkpt_dir
        # self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.gamma = gamma
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(self.input_shape[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(self.input_shape)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, 128)
        self.V = nn.Linear(128, 1)
        self.A = nn.Linear(128, self.n_actions)

        #self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        #self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def calculate_conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0], -1)
        flat1 = F.relu(self.fc1(conv_state))
        flat2 = F.relu(self.fc2(flat1))

        V = self.V(flat2)
        A = self.A(flat2)

        return V, A

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
   
        
    def act(self, state, epsilon):
        if random.uniform(0,1) <= epsilon:
            action = random.randrange(self.env.action_space.n)
        else:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0),requires_grad=True)
            # TODO: Given state, you should write code to get the Q value and chosen action
            _, q_values = self.forward(state) 
            action = torch.argmax(q_values).item()
            
        return action

    def copy_from(self, target):
        self.load_state_dict(target.state_dict())

def replace_target_network(self):
    if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
        self.q_next.load_state_dict(self.q_eval.state_dict())
 
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
    V_s, A_s = model.forward(state)
    V_s_, A_s_ = target_model.forward(next_state)

    indices = np.arange(batch_size)

    q_pred = torch.add(V_s,(A_s - A_s.mean(dim=1, keepdim=True)))[indices, action]
    q_next = torch.add(V_s_,(A_s_ - A_s_.mean(dim=1, keepdim=True))).max(dim=1)[0]

    # print(state_q_values,action.unsqueeze(1))
    # Find selected action's q_value
    #selected_q_value = model.forward(state)[indices, action]
    #q_next = target_model.forward(next_state).max(dim=1)[0]
   
    # Use Bellman function to find expected q value
    expected_q_value = reward + gamma * q_next * (1 - done)

    # Calc loss with expected_q_value and q_value
    loss = (q_pred - expected_q_value.detach()).pow(2).mean()
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
