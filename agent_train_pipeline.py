from agent import *
import itertools
import concurrent.futures
import pickle



# hyperparams to explore
hyperparams = {
    'MAX_MEMORY'      : [50_000, 100_000, 200_000],
    'BATCH_SIZE'      : [1000, 5000],
    'ALPHA'           : [0.0005, 0.001, 0.005],
    'GAMMA'           : [0.7, 0.8, 0.9],
    'EPSILON'         : [0.3, 0.4, 0.5],    
    'EPSILON_FLOOR'   : [0.0, 0.04],          # minimal epsilon value after decay
    'EPSILON_LIN_DEC' : [True, False],        # epsilon decay is linear or exp
    'EPSILON_DEC_LIM' : [70, 80, 90],         # number of games until minimal epsilon
}

# print(hyperparams.items())
total = 1
for k, v in hyperparams.items():
    total *= len(v)
print(f"Total number of model instances to explore: {total}")

# Generate all combinations using itertools.product
k, v = zip(*hyperparams.items())  
combinations = list(itertools.product(*v))

results = {}

# Iterate over each combination and feed it to the model
max_workers = 16
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    for i in range(0, max_workers, max_workers):           # FIXME len(combinations)
        comb_batch = combinations[i:i+max_workers] 
        print(f"Progress: {i}:{i+len(comb_batch)}/{total}")
        # convert each comb to dict ({'MAX_MEMORY' : val, ...})
        hyperparam_comb = [dict(zip(k, h_comb)) for h_comb in comb_batch] 

        # init list of AgentTrainers
        agent_trainers = []
        for h_comb in hyperparam_comb:
            agent_trainers.append(
                AgentTrainer(
                    deque_len=h_comb['MAX_MEMORY'], 
                    train_batch_size=h_comb['BATCH_SIZE'], 
                    alpha=h_comb['ALPHA'], 
                    gamma=h_comb['GAMMA'], 
                    epsilon=h_comb['EPSILON'], 
                    eps_floor=h_comb['EPSILON_FLOOR'], 
                    eps_lin_decay=h_comb['EPSILON_LIN_DEC'], 
                    eps_dec_lim=h_comb['EPSILON_DEC_LIM']
                )
            )
        
        # Submit tasks to the executor --> run max_workers train() in parallel
        # returned dict is stored in tasks[i].result()
        tasks = {executor.submit(a_t.train): a_t for a_t in agent_trainers}
        
        # Wait for all tasks to finish and get the results
        for finished_trainer in concurrent.futures.as_completed(tasks):
            agent_t = tasks[finished_trainer]  # fetch agent_trainer corressponding to result from thread pool
            try:
                result = finished_trainer.result()  # save results
                print(f"  {str(agent_t)} completed")
                results[str(agent_t)] = result
            except Exception as e:
                print(f"  {str(agent_t)} generated an exception: {e}")

print("All results saved to results.pkl")
with open('results.pkl', 'wb') as pickle_file:
    pickle.dump(results, pickle_file)
