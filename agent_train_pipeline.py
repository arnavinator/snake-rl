from agent import *

results = {}

agent_trainer = AgentTrainer()
results["param1"] = agent_trainer.train(interactive_mode=False)

# agent_trainer = AgentTrainer()
# results["param2"] = agent_trainer.train(interactive_mode=False)

print(results)

# hyperparams to explore
# MAX_MEMORY = [50_000, 100_000, 200_000]
# BATCH_SIZE = [500, 1000, 2000, 5000]
# ALPHA = []

# deque_len=MAX_MEMORY, train_batch_size=BATCH_SIZE, 
# alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, 
# eps_floor=EPSILON_FLOOR, eps_lin_decay=EPSILON_LIN_DEC, eps_dec_lim=EPSILON_DEC_LIM,
# block_sz=BLOCK_SIZE, speed_fps=SPEED_FPS