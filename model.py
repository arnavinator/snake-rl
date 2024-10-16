import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class QNet(nn.Module):           
    def __init__(self, num_blocks_visibility):
        super().__init__()
        S1_hidden_size = 128
        S2_h = num_blocks_visibility+1
        S2_w = num_blocks_visibility*2+1
        S2_conv_out_ch = 3

        # fully connected subnetwork to process 1D Compass Snake Dir and Compass Food Dir
        self.S1_linear_1 = nn.Linear(4, S1_hidden_size)   # (B, 10) --> (B, 128)
        self.S1_linear_2 = nn.Linear(S1_hidden_size, S1_hidden_size)  # (B, 128) --> (B, 128)
        self.S1_dropout = nn.Dropout(p=0.1)
        # conv2d subnetwork to process 2D Snake POV for obstacles
        self.S2_conv2d_1 = nn.Conv2d(in_channels=1, out_channels=S2_conv_out_ch, kernel_size=2, stride=1)  # (B, 1, h, w) --> (B, 3, h-1, w-1)
        self.S2_conv2d_2 = nn.Conv2d(in_channels=S2_conv_out_ch, out_channels=S2_conv_out_ch, kernel_size=2, stride=1)  # (B, 3, h-1, w-1) --> (B, 3, h-2, w-2)
        self.S2_linear   = nn.Linear(((S2_h-2)*(S2_w-2)*S2_conv_out_ch), (S2_h*S2_w))
        self.S2_dropout = nn.Dropout(p=0.1)
        # fully connected subnetwork combining prev subnetworks
        self.S3_linear_1 = nn.Linear(S1_hidden_size+(S2_h*S2_w), 80)  # (B, 128+30) --> (B, 80)
        self.S3_linear_2 = nn.Linear(80, 30)  # (B, 80) --> (B, 30)
        self.S3_linear_3 = nn.Linear(30, 3)  # (B, 30) --> (B, 3)


    def forward(self, state):
        if len(state) == 2:  # if B=1, so only one compass_s and one visibility_s
            compass_s, visbility_s = state
            compass_s = torch.unsqueeze(torch.tensor(compass_s, dtype=torch.float), 0)
            visbility_s = torch.unsqueeze(torch.tensor(visbility_s, dtype=torch.float), 0)
        else: # B=len(state), need to extract tuples
            compass_s, visbility_s = zip(*state)  
            compass_s = torch.tensor(np.array(compass_s), dtype=torch.float)
            visbility_s = torch.tensor(np.array(visbility_s), dtype=torch.float)

        # S1
        S1 = F.relu(self.S1_linear_1(compass_s))          # (B,18) --> (B,128)
        S1 = F.relu(self.S1_linear_2(S1))                 # (B,128) --> (B,128)
        S1 = self.S1_dropout(S1)

        # S2 
        b, h, w = visbility_s.shape
        visbility_s_ = visbility_s.view(b, 1, h, w)     # (B, h, w) --> (B, 1, h, w)
        S2 = self.S2_conv2d_1(visbility_s_)             # (B, 1, h, w) --> (B, 2, h-1, w-1)
        S2 = self.S2_conv2d_2(S2)                       # (B, 2, h-1, w-1) --> (B, 2, h-2, w-2)
        S2_flatten = S2.view(b, -1)                     # (B, 2, h', w') --> (B, 2*h'*w')
        S2_flatten = F.relu(self.S2_linear(S2_flatten)) # (B, 2*h'*w') --> (B, h*w)
        S2_flatten = S2_flatten + visbility_s.view(b, -1)           # residual connection
        S2_flatten = self.S2_dropout(S2_flatten)                    # dropout
        # S3
        S12_merge = torch.cat((S1, S2_flatten), dim=1)  # {(B,128), (B, 2*h'*w')} --> (B, 128+2*h'*w')
        S3 = F.relu(self.S3_linear_1(S12_merge))        # (B, 128+2*h'*w') --> (B, 80)
        S3 = F.relu(self.S3_linear_2(S3))               # (B, 80) --> (B, 30)
        S3 = self.S3_linear_3(S3)                       # (B, 30) --> (B, 3)
        return S3

    # save if score > record
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, alpha, alpha_decay, gamma):
        self.alpha = alpha
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.alpha)
        if alpha_decay:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=90, gamma=0.5)
        else:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.5) # not to be reached
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        self.model.train()

        action = torch.tensor(np.array(action), dtype=torch.int)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        # ensure inputs are consistenty indexable 
        # convert tensors from (x)-->(1,x) if needed
        if len(action.shape) == 1:
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )  # len-1 tuple

        # Q(s,a) == pred[action]
        pred = self.model(state)   # (B, 3)

        target = pred.clone()
        for idx in range(len(done)):
            # Q(s,a) <- Q(s,a) + alpha*[R + gamma*max_a'(Q(s',a')) - Q(s,a)]
            # for DQN, alpha=1 (compensated by optimizer learning rate)
            # Q(s,a) <- R + gamma*max_a'(Q(s',a')) 
            if not done[idx]:
                # next_state was chosen via max_a', so max_a'(Q(s',a')) == max(Q(next_state))
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            else:
                Q_new = reward[idx]  # no next action exists

            # update Q(s,a)
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        self.loss = self.criterion(target, pred)  
        self.loss.backward() 
        self.optimizer.step()
    
    def step_alpha_scheduler(self):
        self.scheduler.step()


