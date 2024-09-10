import torch
import random
import numpy as np
# deque: linked list, so O(1) for LHS pop, O(n) for index
# list: dynamic array, so O(n) for LHS pop, O(1) for index
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

# # set random seed
# seed = 1298734123
# random.seed(seed)            
# np.random.seed(seed)      
# torch.manual_seed(seed) 

# Agent() Training params
MAX_MEMORY      = 100_000
BATCH_SIZE      = 1000

# Agent() Q-learning params   
ALPHA           = 0.001  # to Adam optimizer for deep-Q-learning... TODO implement ALPHA decay? V3.x
GAMMA           = 0.9         
EPSILON         = 0.4    
EPSILON_FLOOR   = 0.00   # minimal epsilon value after decay
EPSILON_LIN_DEC = True   # epsilon decay is linear or exp
EPSILON_DEC_LIM = 80     # number of games until minimal epsilon

# SnakeGameAI() params
BLOCK_SIZE  = 20         # size of one unit of movement (visual, non-functional) 
# train_short_memory() + remember() latency = ~0.001s on machine 
#  --> allow for 0.005s latency per frame 
SPEED_FPS   = 200  

class Agent:
    def __init__(self, deque_len, train_batch_size, alpha, gamma, epsilon, eps_floor, eps_lin_decay, eps_dec_lim):
        self.n_games = 0
        self.memory = deque(maxlen=deque_len)
        self.IN_STATE_LEN = 11   
        self.model = Linear_QNet(input_size=self.IN_STATE_LEN, hidden_size=256, output_size=3)
        self.trainer = QTrainer(self.model, alpha=alpha, gamma=gamma)
        self.train_batch_size = train_batch_size
        self.eps_lin_decay = eps_lin_decay  # bool for lin decay or exp decay
        self.n_games_eps_decay = eps_dec_lim
        if self.eps_lin_decay:
            self.epsilon = epsilon 
            # epsilon -= eps_step --> lin decay to eps_floor% after n_games_eps_decay games
            self.eps_step = (epsilon - eps_floor) / self.n_games_eps_decay   
        else:
            self.epsilon = torch.tensor(epsilon, dtype=torch.float16)
            # epsilon *= eps_step --> exp decay to eps_floor% after n_games_eps_decay games
            self.eps_step = torch.tensor(np.e**(np.log(eps_floor/epsilon) / self.n_games_eps_decay), dtype=torch.float16)

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
        
        
    # popleft if deque_len is reached --> only remember deque_len frames
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # deque of tuples

    # train once every frame
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    # train BATCH_SIZE when game_over
    def train_long_memory(self):
        # if len(memory) > self.train_batch_size, take rand sample for train_long
        if len(self.memory) > self.train_batch_size: 
            mini_sample = random.sample(self.memory, self.train_batch_size) # list of tuples
        else:
            mini_sample = self.memory
        
        # group together all (states, ..., dones) from mini_sample
        #   into (BATCH_SIZE, {original len})
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        # train on BATCH_SIZE sample. 
        #   convert list of nparray to single nparray, faster tensor conversion
        self.trainer.train_step(np.array(states), 
                                np.array(actions),
                                np.array(rewards), 
                                np.array(next_states), 
                                dones)

    # called once every game over
    def update_epsilon(self):
        # epsilon-rand: linear/exp decay to eps_floor% after n_games_eps_decay games
        if (self.n_games <= self.n_games_eps_decay):
            if self.eps_lin_decay: # linear decay 
                self.epsilon -= self.eps_step  
            else:
                self.epsilon *= self.eps_step

    # epsilon-greedy action 
    def get_action(self, state):        
        if (torch.rand(1, dtype=torch.float16) < self.epsilon).item():  # exploration
            move = random.randint(0, 2)
        else:  # exploitation 
            state0 = torch.tensor(state, dtype=torch.float)
            # state (11-dim) --> model --> Q(s) (3-dim)
            prediction = self.model(state0)
            # pick action yielding highest Q 
            move = torch.argmax(prediction).item()
        
        final_move = [0,0,0] 
        final_move[move] = 1 # one-hot action

        return final_move

class AgentTrainer:
    def __init__(self, deque_len=MAX_MEMORY, train_batch_size=BATCH_SIZE, 
                    alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, 
                    eps_floor=EPSILON_FLOOR, eps_lin_decay=EPSILON_LIN_DEC, eps_dec_lim=EPSILON_DEC_LIM,
                    block_sz=BLOCK_SIZE, speed_fps=SPEED_FPS):
        # set random seed
        seed = 1298734123
        random.seed(seed)            
        np.random.seed(seed)      
        torch.manual_seed(seed) 

        self.track_scores = []
        self.track_ma_scores = []
        self.track_efficiency = []
        self.record = 0
        self.total_num_frames = 0
        self.model_loss = []
        self.agent = Agent(deque_len, train_batch_size, alpha, gamma, epsilon, 
                           eps_floor, eps_lin_decay, eps_dec_lim)
        self.game = SnakeGameAI(BLOCK_SIZE=block_sz, SPEED=speed_fps)

        # input_size should match for QNet
        assert len(self.agent.get_state(self.game)) == self.agent.IN_STATE_LEN  

    def find_frame_goal(self): 
        m_x_dist = np.abs(self.game.food.x - self.game.head.x)//self.game.BLOCK_SIZE # manhattan x-dist to food
        m_y_dist = np.abs(self.game.food.y - self.game.head.y)//self.game.BLOCK_SIZE # manhattan y-dist to food
        golden_frame_cnt = m_x_dist + m_y_dist
        return golden_frame_cnt

    def train(self, interactive_mode=True):   # interactive_mode enables live plot, model save, print statements
        # init current_state
        current_state = self.agent.get_state(self.game)

        # init for track_efficiency 
        prev_score = 0
        frame_checkpoint = 0
        ideal_frame_cnt = self.find_frame_goal()

        while (not self.game.quit) and (self.agent.n_games < 120):
            # get current state
            if current_state is None:
                current_state = self.agent.get_state(self.game)

            # Q-learning epsilon-greedy move prediction
            new_move = self.agent.get_action(current_state)

            # perform move and get new state
            reward, done, score = self.game.play_step(new_move, interactive_mode)
            new_state = self.agent.get_state(self.game)

            # train short memory (every frame)
            self.agent.train_short_memory(current_state, new_move, reward, new_state, done)

            # remember, for train_long_memory()
            self.agent.remember(current_state, new_move, reward, new_state, done)

            # for next iter in game (no need to recalc current_state)
            current_state = new_state

            # update track_efficiency if snake ate food
            if prev_score != score:
                prev_score = score
                frame_checkpoint = self.total_num_frames
                golden_frame_cnt = self.find_frame_goal()
                ideal_frame_cnt += golden_frame_cnt

            self.total_num_frames += 1
    
            if done:  # aka game_over
                # train long memory, plot result
                self.game.reset()
                self.agent.n_games += 1

                self.agent.update_epsilon()

                self.agent.train_long_memory()

                if score > self.record:
                    self.record = score
                    if interactive_mode:
                        self.agent.model.save()

                loss = self.agent.trainer.loss.item()
                if interactive_mode:
                    print('Game:', self.agent.n_games, '\tScore:', score, '\tRecord:', self.record, '\tReplay Loss:', round(loss, 4))
                
                # --- update plot metrics --- 
                self.track_scores.append(score)
                score_slice = self.track_scores[-5:]  # 5-game simple moving avg
                self.track_ma_scores.append(sum(score_slice) / len(score_slice))
                self.model_loss.append(loss)
                if score == 0 or (frame_checkpoint-start_frame_count == 0):  # clamp in case food spawn at head --> divby0
                    efficiency = 0
                else:
                    # num "extra" frames to complete objective, discounted for score (difficulty)
                    #   only consider "success" runs, final failed attempt for food excluded
                    success_ideal_frame_cnt = ideal_frame_cnt - golden_frame_cnt 
                    efficiency = (success_ideal_frame_cnt + score) / (frame_checkpoint - start_frame_count) # 0 (bad) --> ~1 (good)
                self.track_efficiency.append(efficiency)

                if interactive_mode:
                    plot(self.track_scores, self.track_ma_scores, self.track_efficiency)

                # for next game
                current_state = None
                start_frame_count = self.total_num_frames
                prev_score = 0
                ideal_frame_cnt = self.find_frame_goal()
        
        # training complete 
        if interactive_mode:
            print("Total Number of Frames:                            ",self.total_num_frames)
            print("Total Number of Games:                             ", self.agent.n_games)
            print("Record Length:                                     ", self.record)
            print("Average Score across Games:                        ", sum(self.track_scores)/len(self.track_scores) )
            print("Mean(Last 20 Games Length):                        ", np.mean(self.track_scores[-20:]) )
            print("Stdev(Last 20 Games Length):                       ", np.std(self.track_scores[-20:]) )
            print("Mean(Efficiency Score(Last 20 Games Length)):      ", np.mean(self.track_efficiency[-20:]) )
            print("Stdev(Efficiency Score(Last 20 Games Length)):     ", np.std(self.track_efficiency[-20:]) )
            print("Mean(QNet Replay Loss(Last 10 games)):             ", np.mean(self.model_loss[-10:]))
            print("Training Over. Close Plot to terminate program.")
            plot(self.track_scores, self.track_ma_scores, self.track_efficiency, show_final=True)
        else:
            results = {}
            results["Total Number of Frames"]                           = self.total_num_frames
            results["Total Number of Games"]                            = self.agent.n_games
            results["Record Length"]                                    = self.record
            results["Average Score across Games"]                       = sum(self.track_scores)/len(self.track_scores)
            results["Mean(Last 20 Games Length)"]                       = np.mean(self.track_scores[-20:])
            results["Stdev(Last 20 Games Length)"]                      = np.std(self.track_scores[-20:])
            results["Mean(Efficiency Score(Last 20 Games Length))"]     = np.mean(self.track_efficiency[-20:])
            results["Stdev(Efficiency Score(Last 20 Games Length))"]    = np.std(self.track_efficiency[-20:])
            # in case want to plot results 
            results["Data: Length per Game"] = self.track_scores
            results["Data: 5-game Length Moving Average"] = self.track_ma_scores
            results["Data: Efficiency Score"] = self.track_efficiency
            results["Data: QNet Replay Loss"] = self.model_loss

            return results



if __name__ == '__main__':
    agent_trainer = AgentTrainer()
    agent_trainer.train(interactive_mode=True)