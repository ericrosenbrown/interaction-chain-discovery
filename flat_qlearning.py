from simple_rl.agents import QLearningAgent, RandomAgent, RMaxAgent
from FlatMDPClass import FlatMDP
from simple_rl.run_experiments import run_agents_on_mdp

# Setup MDP.
mdp = FlatMDP(width=5, height=5, init_loc=(0, 0), goal_locs=[(3, 3)], gamma=0.95, slip_prob=0.05)

# Setup Agents.
ql_agent = QLearningAgent(actions=mdp.get_actions())
#rmax_agent = RMaxAgent(actions=mdp.get_actions())
rand_agent = RandomAgent(actions=mdp.get_actions())

# Run experiment and make plot.
run_agents_on_mdp([ql_agent, rand_agent], mdp, instances=5, episodes=200, steps=100)
