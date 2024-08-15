import torch
import random
import numpy as np
# deque: linked list, so O(1) for LHS pop, O(n) for index
# list: dynamic array, so O(n) for LHS pop, O(1) for index
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

# Training params
MAX_MEMORY      = 100_000
BATCH_SIZE      = 1000
IN_STATE_LEN    = 11

# Q-learning params
LR              = 0.001  # to Adam optimizer
ALPHA           = 1      # unused for Deep-Q-learning, TODO v2.x
GAMMA           = 0.9         
EPSILON         = 0      # unused, currently hard-coded 

# SnakeGameAI params
BLOCK_SIZE  = 20
SPEED       = 40

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0            # set in self.get_action()
        self.gamma = GAMMA         
        # if deque len exceeds maxlen, then additional append will popleft()
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(input_size=IN_STATE_LEN, hidden_size=256, output_size=3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [ 
            # snake only finds out danger one block away... TODO V2.x  
            # Danger straight 
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction (only 1 is true)
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location (only 1-2 are true)
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)   # convert T/F list to 0/1s
        
        
    # popleft if MAX_MEMORY is reached --> only remember MAX_MEMORY frames
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # deque of tuples

    # train once every frame
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    # train BATCH_SIZE when game_over
    def train_long_memory(self):
        # if len(memory) > BATCH_SIZE, take rand sample for train_long
        if len(self.memory) > BATCH_SIZE: 
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        
        # group together all (states, ..., dones) from mini_sample
        #   into (BATCH_SIZE, {original len})
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        # train on BATCH_SIZE sample
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    # epsilon-greedy action 
    def get_action(self, state):
        # epsilon-rand: 40% rand linear decay to 0% rand after 
        # ...nonlin decay instead? # TODO V2.x 
        self.epsilon = 80 - self.n_games  
        final_move = [0,0,0] 
        if random.randint(0, 200) < self.epsilon:  # exploration
            move = random.randint(0, 2)
            final_move[move] = 1 # one-hot action
        else:  # exploitation 
            state0 = torch.tensor(state, dtype=torch.float)
            # state (11-dim) --> model --> Q(s) (3-dim)
            prediction = self.model(state0)
            # pick action yielding highest Q 
            move = torch.argmax(prediction).item()
            final_move[move] = 1 # one-hot action

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI(BLOCK_SIZE=BLOCK_SIZE, SPEED=SPEED)

    # input_size should match for QNet
    assert len(agent.get_state(game)) == IN_STATE_LEN  

    while not game.quit:
        # get current state
        # ... should just remember from prev iter... # FIXME V2.x
        current_state = agent.get_state(game)

        # Q-learning epsilon-greedy move prediction
        new_move = agent.get_action(current_state)

        # perform move and get new state
        reward, done, score = game.play_step(new_move)
        new_state = agent.get_state(game)

        # train short memory (every frame)
        agent.train_short_memory(current_state, new_move, reward, new_state, done)

        # remember, for train_long_memory()
        agent.remember(current_state, new_move, reward, new_state, done)

        if done:  # aka game_over
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            # TODO V2.x - instead of train from replay every game, do it 
            # min_iter(every game, 50 iters)... depends on train latency 
            # OR could have train in parallel, newest Q-Net replace old once done
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
    
    print("Training Over. Close Figure 1 to terminate program")
    plot(plot_scores, plot_mean_scores, show_final=True)


if __name__ == '__main__':
    train()