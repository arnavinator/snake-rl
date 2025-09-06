import torch
import random
import numpy as np
# deque: linked list, so O(1) for LHS pop, O(n) for index
# list: dynamic array, so O(n) for LHS pop, O(1) for index
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import QNet, QTrainer
from helper import plot

# Agent() Training params
MAX_MEMORY      = 10_000
BATCH_SIZE      = 512

# Agent() Q-learning params   
ALPHA           = 0.001  # to Adam optimizer for deep-Q-learning
ALPHA_DECAY     = False
GAMMA           = 0.9         
EPSILON         = 0.4    
EPSILON_FLOOR   = 0.00   # minimal epsilon value after decay
EPSILON_LIN_DEC = True   # epsilon decay is linear or exp
EPSILON_DEC_LIM = 80     # number of games until minimal epsilon
PRI_REPLAY_EN   = False  # priority replay buffer at end of every episode

# SnakeGameAI() params
BLOCK_SIZE  = 20         # size of one unit of movement (visual, non-functional) 
# train_short_memory() + remember() latency = ~0.001s on machine 
#  --> allow for 0.005s latency per frame 
SPEED_FPS   = 200

class Agent:
    def __init__(self, deque_len, train_batch_size, alpha, alpha_decay, gamma, epsilon, eps_floor, eps_lin_decay, eps_dec_lim):
        self.n_games = 0
        self.deque_len = deque_len
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon_full = epsilon 
        self.eps_floor = eps_floor
        self.train_batch_size = train_batch_size
        self.eps_lin_decay = eps_lin_decay  # bool for lin decay or exp decay
        self.n_games_eps_decay = eps_dec_lim

        self.num_blocks_visibility = 3    # what is the field of view for snake head obstacle detection

        self.memory = deque(maxlen=deque_len)
        self.model = QNet(self.num_blocks_visibility)
        self.trainer = QTrainer(self.model, alpha=alpha, alpha_decay=alpha_decay, gamma=gamma)
        
        if self.eps_lin_decay:
            self.epsilon = epsilon 
            # epsilon -= eps_step --> lin decay to eps_floor% after n_games_eps_decay games
            self.eps_step = (epsilon - eps_floor) / self.n_games_eps_decay   
        else:
            self.epsilon = torch.tensor(epsilon, dtype=torch.float16)
            # epsilon *= eps_step --> exp decay to eps_floor% after n_games_eps_decay games
            self.eps_step = torch.tensor(np.e**(np.log((eps_floor+1e-5)/epsilon) / self.n_games_eps_decay), dtype=torch.float16)

    def __repr__(self):
        return f"Agent(MAX_MEMORY={self.deque_len}, BATCH_SIZE={self.train_batch_size}, ALPHA={self.alpha}, ALPHA_DECAY={self.alpha_decay}, " \
                     f"GAMMA={self.gamma}, EPSILON={self.epsilon_full}, EPSILON_FLOOR={self.eps_floor}, " \
                     f"EPSILON_LIN_DEC={self.eps_lin_decay}, EPSILON_DEC_LIM={self.n_games_eps_decay})"
    
    def danger_dist(self, game, relative_danger_dir, num_blocks_visibility):
        # neuron input between -0.75 to 0.75, where
        # -0.75      --> no danger               (lowest priority)
        # -0.75+step --> danger is dist_from_curr units away
        # ...
        # 0.75       --> danger is 1 unit away   (highest priority)

        # initalize possible Q-Net input values
        danger_vals = np.linspace(0.75, -0.75, num=(num_blocks_visibility+1))

        # translate relative_danger_dir to abs_danger_dir (compass direction)
        curr_comp_dir = game.direction
        clockwise_dir = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        curr_idx = clockwise_dir.index(curr_comp_dir)
        if relative_danger_dir == "S":    # compass dir of danger Straight relative to curr_comp_dir
            abs_danger_dir = curr_comp_dir
        elif relative_danger_dir == "R":  # compass dir of danger Right relative to curr_comp_dir
            abs_danger_dir = clockwise_dir[((curr_idx + 1) % 4)]
        elif relative_danger_dir == "L":  # compass dir of danger Left relative to curr_comp_dir
            abs_danger_dir = clockwise_dir[((curr_idx - 1) % 4)]
        
        head = game.snake[0]

        # assign danger_vals
        for i in range(1,num_blocks_visibility+1):
            if abs_danger_dir == Direction.LEFT:
                danger_point = Point(head.x - BLOCK_SIZE*i, head.y)
            elif abs_danger_dir == Direction.RIGHT:
                danger_point = Point(head.x + BLOCK_SIZE*i, head.y)
            elif abs_danger_dir == Direction.UP:
                danger_point = Point(head.x, head.y - BLOCK_SIZE*i)
            elif abs_danger_dir == Direction.DOWN:
                danger_point = Point(head.x, head.y + BLOCK_SIZE*i)

            if game.is_collision(danger_point):
                return danger_vals[i-1]
        return danger_vals[-1]
   
    def danger_view(self, game):
        nbv = self.num_blocks_visibility
        BS = game.BLOCK_SIZE

        h = nbv+1
        w = nbv*2+1  
        danger = np.zeros((h,w), dtype=int)
        head = game.snake[0]

        for i in range(h):
            for j in range(w):
                if game.direction == Direction.UP:
                    x = head.x + (j-nbv)*BS
                    y = head.y + (i-nbv)*BS
                elif game.direction == Direction.DOWN:
                    x = head.x + (nbv-j)*BS
                    y = head.y + (nbv-i)*BS
                elif game.direction == Direction.RIGHT:
                    x = head.x + (nbv-i)*BS
                    y = head.y + (j-nbv)*BS
                elif game.direction == Direction.LEFT:
                    x = head.x + (i-nbv)*BS
                    y = head.y + (nbv-j)*BS
                danger[i, j] = game.is_collision(Point(x,y))
        
        # for consistency, self.head itself should be collision
        danger[-1, nbv] = 1  
        
        return danger

    def food_view(self, game):
        # only 1 comp dir is True for snake head dir
        snake_d = [   # N, E, S, W
            game.direction == Direction.UP,       # snake head facing N
            game.direction == Direction.RIGHT,    # snake head facing E
            game.direction == Direction.DOWN,     # snake head facing S
            game.direction == Direction.LEFT      # snake head facing W
        ]
        snake_d = np.array(snake_d, dtype=int)
        snake_comp_idx = np.where(snake_d == 1)[0][0]  # idx indicating N/E/S/W

        # up to 2 comp dir is True for food dir
        # find food dir, check if clear path to food
        # -1: no food, 0: food but obstacle, 1: food and no obstacle
        food_d = np.array([-1, -1, -1, -1])   # N, E, S, W
        if game.food.y < game.head.y:       # check N
            food_d[0] = 1  # assume no obstacle
            yt = game.head.y
            while yt != game.food.y:
                yt -= game.BLOCK_SIZE
                if game.is_collision(Point(game.head.x, yt)):
                    food_d[0] = 0  # obstacle found
                    break
        elif game.food.y > game.head.y:     # check S
            food_d[2] = 1  # assume no obstacle
            yt = game.head.y
            while yt != game.food.y:
                yt += game.BLOCK_SIZE
                if game.is_collision(Point(game.head.x, yt)):
                    food_d[2] = 0  # obstacle found
                    break

        if game.food.x > game.head.x:       # check E
            food_d[1] = 1  # assume no obstacle
            xt = game.head.x
            while xt != game.food.x:
                xt += game.BLOCK_SIZE
                if game.is_collision(Point(xt, game.head.y)):
                    food_d[1] = 0  # obstacle found
                    break
        elif game.food.x < game.head.x:     # check W
            food_d[3] = 1  # assume no obstacle
            xt = game.head.x
            while xt != game.food.x:
                xt -= game.BLOCK_SIZE
                if game.is_collision(Point(xt, game.head.y)):
                    food_d[3] = 0  # obstacle found
                    break

        food_comp_list = np.where(food_d > -1)[0]   # list of idx indicating N/E/S/W

        # determine if food is Straight/Backwards and/or Left/Right 
        #    and combine with knowledge of obstacle ahead
        # -1: no food, 0: food but obstacle, 1: food and no obstacle
        food_relative_dir = np.array([-1,-1,-1,-1]) # Straight, Backwards, Left, Right

        for food_idx in food_comp_list:
            h = snake_comp_idx   # head idx
            f = food_idx         # food idx

            value = food_d[f]    # obstacle or no obstacle

            # compare compass directions of food to determine relative direction
            if h == f:
                food_relative_dir[0] = value  # food is relative Straight
            elif (h-f)%4 == 2:
                food_relative_dir[1] = value  # food is relative Back

            if (h-f)%4 == 1:
                food_relative_dir[2] = value  # food is relative Left
            elif (f-h)%4 == 1:
                food_relative_dir[3] = value  # food is relative Right
        
        return food_relative_dir

            
    def get_state(self, game):

        # new_state = np.array([ 
        #     # NEW STUFF
        #     game.score / 100,   # snake len
        #     np.abs(game.food.x - game.head.x) / game.w,   # dist from food in x
        #     np.abs(game.food.y - game.head.y) / game.h,
        #     self.danger_dist(game=game, relative_danger_dir="S", num_blocks_visibility=10),
        #     self.danger_dist(game=game, relative_danger_dir="L", num_blocks_visibility=10),
        #     self.danger_dist(game=game, relative_danger_dir="R", num_blocks_visibility=10),
        # ], dtype=float)

        compass_s = self.food_view(game)

        # compass_s = np.concat((compass_s, new_state), axis=0)    

        visisbility_s = self.danger_view(game)  # Danger view --- 2D array

        return (compass_s, visisbility_s) 
        
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
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    # train BATCH_SIZE when game_over, with priority to end frames
    def train_priority_long_memory(self, n_end_frames, pri_end_rewards, positive_reward):
        end_frames = [self.memory[i] for i in range(-n_end_frames, 0)]  # make sure last few frames are used
        # if len(memory) > self.train_batch_size-n_end_frames, take rand sample for train_long
        if len(self.memory) > (self.train_batch_size-n_end_frames): 
            mini_sample = random.sample(self.memory, (self.train_batch_size-n_end_frames)) # list of tuples
            mini_sample += end_frames   # concatenate to make mini_sample
        else:
            mini_sample = self.memory
        
        # group together all (states, ..., dones) from mini_sample
        #   into (BATCH_SIZE, {original len})
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        # near end-of-game rewards (snake bites itself) slightly penalized!
        if pri_end_rewards:
            rewards = np.array(rewards)
            r_tail = rewards[-n_end_frames:]
            # check that none of the last n_end_frames was reward of any kind
            check_reward = np.where(r_tail != 0)[0]
            if check_reward.size == 0: # no reward here in window
                idx = None  
                steps = n_end_frames
            else:
                idx = check_reward[-1]  # most recent reward
                steps = n_end_frames - (idx+1)

            # modify rewards, only if room to modify
            if steps != 0:
                # values to between 0 and +/-10, exponentially decr over steps intervals 
                exponential_values = np.logspace(-1, 0, num=steps)
                if positive_reward:
                    scaled_values = exponential_values * 10
                else:
                    scaled_values = exponential_values * -10
                # modify rewards
                rewards = np.concat((rewards[:-steps], scaled_values), axis=0)   

            # now let's store these modified rewards back into memory (so can be resampled) 
            # end-of-game entries are at end of mini sample 
            #   --> pop them off, rezip end of mini sample with revised rewards, update memory      
            for i in range(steps):  # remove the entries we want to revise
                self.memory.pop()
            self.memory += deque(zip(states[-steps:], actions[-steps:], rewards[-steps:], next_states[-steps:], dones[-steps:]))

        # train on BATCH_SIZE sample. 
        #   convert list of nparray to single nparray, faster tensor conversion
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    # called once every game over
    def update_epsilon(self):
        # epsilon-rand: linear/exp decay to eps_floor% after n_games_eps_decay games
        if (self.n_games <= self.n_games_eps_decay):
            if self.eps_lin_decay: # linear decay 
                self.epsilon -= self.eps_step  
            else:
                self.epsilon *= self.eps_step
        else:
            self.epsilon = 0

    # epsilon-greedy action 
    def get_action(self, state):        
        if (torch.rand(1, dtype=torch.float16) < self.epsilon).item():  # exploration
            move = random.randint(0, 2)
        else:  # exploitation 
            # state (11-dim) --> model --> Q(s) (3-dim)
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(state)
            # pick action yielding highest Q 
            move = torch.argmax(prediction).item()
        
        final_move = [0,0,0] 
        final_move[move] = 1 # one-hot action

        return final_move

class AgentTrainer:
    def __init__(self, deque_len=MAX_MEMORY, train_batch_size=BATCH_SIZE, 
                    alpha=ALPHA, alpha_decay=ALPHA_DECAY, gamma=GAMMA, epsilon=EPSILON, 
                    eps_floor=EPSILON_FLOOR, eps_lin_decay=EPSILON_LIN_DEC, eps_dec_lim=EPSILON_DEC_LIM,
                    pri_replay_en=PRI_REPLAY_EN, block_sz=BLOCK_SIZE, speed_fps=SPEED_FPS, interactive_mode=False):
        # set random seed
        seed = 1298734123
        random.seed(seed)            
        np.random.seed(seed)      
        torch.manual_seed(seed) 

        self.interactive_mode = interactive_mode  # enables live plot, model save, print statements
        self.track_scores = []
        self.track_ma_scores = []
        self.track_efficiency = []
        self.record = 0
        self.total_num_frames = 0
        self.model_loss = []
        self.agent = Agent(deque_len, train_batch_size, alpha, alpha_decay, gamma, epsilon, 
                           eps_floor, eps_lin_decay, eps_dec_lim)
        self.game = SnakeGameAI(BLOCK_SIZE=block_sz, SPEED=speed_fps, interactive_mode=interactive_mode)
        self.pri_replay_en = pri_replay_en
    
    def __repr__(self):
        return str(self.agent)[:-1] + f", PRI_REPLAY_EN={self.pri_replay_en})"

    def find_frame_goal(self): 
        m_x_dist = np.abs(self.game.food.x - self.game.head.x)//self.game.BLOCK_SIZE # manhattan x-dist to food
        m_y_dist = np.abs(self.game.food.y - self.game.head.y)//self.game.BLOCK_SIZE # manhattan y-dist to food
        golden_frame_cnt = m_x_dist + m_y_dist
        return golden_frame_cnt

    def train(self):    
        # init current_state
        current_state = self.agent.get_state(self.game)

        # init for track_efficiency 
        prev_score = 0
        frame_checkpoint = 0
        start_frame_count = 0
        ideal_frame_cnt = self.find_frame_goal()

        while (not self.game.quit) and (self.agent.n_games < 120):
            # get current state
            if current_state is None:
                current_state = self.agent.get_state(self.game)

            # Q-learning epsilon-greedy move prediction
            new_move = self.agent.get_action(current_state)

            # perform move and get new state
            reward, done, score = self.game.play_step(new_move)
            new_state = self.agent.get_state(self.game)

            # train short memory (every frame) # deprecated
            # self.agent.train_short_memory(current_state, new_move, reward, new_state, done)

            # remember, for train_long_memory()
            self.agent.remember(current_state, new_move, reward, new_state, done)

            if (self.total_num_frames+2) % 4 == 0:
                self.agent.train_long_memory()

            # for next iter in game (no need to recalc current_state)
            current_state = new_state

            # update track_efficiency if snake ate food
            if prev_score != score:
                prev_score = score
                frame_checkpoint = self.total_num_frames
                golden_frame_cnt = self.find_frame_goal()
                ideal_frame_cnt += golden_frame_cnt
                if self.pri_replay_en:  
                    n_end_frames = 7
                    if self.agent.n_games >= 35:   # once snake starts biting itself, penalize end states
                        self.agent.train_priority_long_memory(n_end_frames=n_end_frames, pri_end_rewards=False, positive_reward=True)


            self.total_num_frames += 1
    
            if done:  # aka game_over
                # train long memory, plot result
                self.game.reset()
                self.agent.n_games += 1

                if score > self.record:
                    self.record = score
                    if self.interactive_mode:
                        self.agent.model.save()

                loss = self.agent.trainer.loss.item()
                if self.interactive_mode:
                    print('Game:', self.agent.n_games, '\tScore:', score, '\tRecord:', self.record, '\tReplay Loss:', round(loss, 4))
                
                if self.pri_replay_en:  
                    n_end_frames = 7
                    if self.agent.n_games < 35:   # once snake starts biting itself, penalize end states
                        self.agent.train_priority_long_memory(n_end_frames=n_end_frames, pri_end_rewards=False, positive_reward=False)
                    else:
                        self.agent.train_priority_long_memory(n_end_frames=n_end_frames, pri_end_rewards=True, positive_reward=False)

                self.agent.trainer.step_alpha_scheduler()
                self.agent.update_epsilon()

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

                if self.interactive_mode:
                    plot(self.track_scores, self.track_ma_scores, self.track_efficiency)

                # for next game
                current_state = None
                start_frame_count = self.total_num_frames
                prev_score = 0
                ideal_frame_cnt = self.find_frame_goal()
        
        # training complete 
        if self.interactive_mode:
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
            # # in case want to plot results 
            # results["Data: Length per Game"] = self.track_scores
            # results["Data: 5-game Length Moving Average"] = self.track_ma_scores
            # results["Data: Efficiency Score"] = self.track_efficiency
            # results["Data: QNet Replay Loss"] = self.model_loss

            return results



if __name__ == '__main__':
    h_comb = {
        'MAX_MEMORY'      : 10_000,    # 10k: remember last ~15-20 games
        'BATCH_SIZE'      : 500,
        'ALPHA'           : 0.001,
        'ALPHA_DECAY'     : True,
        'GAMMA'           : 0.7,
        'EPSILON'         : 0.4,    
        'EPSILON_FLOOR'   : 0.0,              # minimal epsilon value after decay
        'EPSILON_LIN_DEC' : False,            # epsilon decay is linear or exp
        'EPSILON_DEC_LIM' : 80,           # number of games until minimal epsilon
        'PRI_REPLAY_EN'   : True,             # priority replay buffer at end of every episode
    }

    agent_trainer = AgentTrainer(
        deque_len=h_comb['MAX_MEMORY'], 
        train_batch_size=h_comb['BATCH_SIZE'], 
        alpha=h_comb['ALPHA'], 
        alpha_decay=h_comb['ALPHA_DECAY'], 
        gamma=h_comb['GAMMA'], 
        epsilon=h_comb['EPSILON'], 
        eps_floor=h_comb['EPSILON_FLOOR'], 
        eps_lin_decay=h_comb['EPSILON_LIN_DEC'], 
        eps_dec_lim=h_comb['EPSILON_DEC_LIM'],
        pri_replay_en=h_comb['PRI_REPLAY_EN'],
        interactive_mode=True)
    
    agent_trainer.train()


