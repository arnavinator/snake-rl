import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):           
    # quantize?...  TODO V2.x
    # maybe 2 hidden layers, with hidden_size smaller? ... TODO V2.x
    # maybe 2 hidden layer, with 1 layer as conv2d? ... TODO V2.x
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x) 
        # no final activation layer... add Softmax? # TODO V2.x 
        return x

    # save if score > record
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.int)
        reward = torch.tensor(reward, dtype=torch.float)
        
        # ensure inputs are consistenty indexable 
        # if parent is train_short_memory(), convert tensors from (x)-->(1,x)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )  # len-1 tuple
        # else parent is train_long_memory(), s.t. tensors are (BATCH_SIZE, x)

        # Q(s,a) == pred[action]
        pred = self.model(state)   # (BATCH_SIZE, 3)

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



