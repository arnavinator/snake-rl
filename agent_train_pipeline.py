from agent import *

results = {}

agent_trainer = AgentTrainer()
results["param1"] = agent_trainer.train(interactive_mode=False)

# agent_trainer = AgentTrainer()
# results["param2"] = agent_trainer.train(interactive_mode=False)

print(results)