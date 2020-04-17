from simple_rl.agents import QLearningAgent, RandomAgent, RMaxAgent
from FlatMDPClass import FlatMDP
from simple_rl.run_experiments import run_agents_on_mdp
import random

from collections import defaultdict

# Setup MDP.
mdp = FlatMDP(width=5, height=5, init_loc=(0, 0), goal_locs=[(3, 3)], gamma=0.95, slip_prob=0.05)

histories = []

for episode in range(100):
    mdp.reset()
    history = [mdp.cur_state]
    for step in range(100):
        mdp.execute_agent_action(random.choice(mdp.get_actions()))
        history.append(mdp.cur_state)
        if mdp.is_goal_state(mdp.cur_state):
            mdp.reset()
    histories.append(history)

#robot_dict[[1,2]['left'] = {'press': 20, 'unpress':0}}
robot_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
u_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
r_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
d_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
l_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
game_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


for episode in histories:
    for i in range(1,len(episode)):
       last_state = episode[i-1]
       cur_state = episode[i]

 
       #Detect button dynamics
       #Button dynamics are either a) being pressed (True) or b) not being pressed
       try:
           robot_dict[(cur_state.xr,cur_state.yr)]['u'][cur_state.u] += 1
           game_dict[(cur_state.xg,cur_state.yg)]['u'][cur_state.u] += 1
           r_dict[cur_state.r]['u'][cur_state.u] += 1
           d_dict[cur_state.d]['u'][cur_state.u] += 1
           l_dict[cur_state.l]['u'][cur_state.u] += 1
       except:
           robot_dict[(cur_state.xr,cur_state.yr)]['u'][cur_state.u] = 1
           game_dict[(cur_state.xg,cur_state.yg)]['u'][cur_state.u] = 1
           r_dict[cur_state.r]['u'][cur_state.u] = 1
           d_dict[cur_state.d]['u'][cur_state.u] = 1
           l_dict[cur_state.l]['u'][cur_state.u] = 1
       try:
           robot_dict[(cur_state.xr,cur_state.yr)]['r'][cur_state.r] += 1
           game_dict[(cur_state.xg,cur_state.yg)]['r'][cur_state.r] += 1
           u_dict[cur_state.u]['r'][cur_state.r] += 1
           d_dict[cur_state.d]['r'][cur_state.r] += 1
           l_dict[cur_state.l]['r'][cur_state.r] += 1
       except:
           robot_dict[(cur_state.xr,cur_state.yr)]['r'][cur_state.r] = 1
           game_dict[(cur_state.xg,cur_state.yg)]['r'][cur_state.r] = 1
           u_dict[cur_state.u]['r'][cur_state.r] = 1
           d_dict[cur_state.d]['r'][cur_state.r] = 1
           l_dict[cur_state.l]['r'][cur_state.r] = 1
       try:
           robot_dict[(cur_state.xr,cur_state.yr)]['d'][cur_state.d] += 1
           game_dict[(cur_state.xg,cur_state.yg)]['d'][cur_state.d] += 1
           u_dict[cur_state.u]['d'][cur_state.d] += 1
           r_dict[cur_state.r]['d'][cur_state.d] += 1
           l_dict[cur_state.l]['d'][cur_state.d] += 1
       except:
           robot_dict[(cur_state.xr,cur_state.yr)]['d'][cur_state.d] = 1
           game_dict[(cur_state.xg,cur_state.yg)]['d'][cur_state.d] = 1
           u_dict[cur_state.u]['d'][cur_state.d] = 1
           r_dict[cur_state.r]['d'][cur_state.d] = 1
           l_dict[cur_state.l]['d'][cur_state.d] = 1
       try:
           robot_dict[(cur_state.xr,cur_state.yr)]['l'][cur_state.l] += 1
           game_dict[(cur_state.xg,cur_state.yg)]['l'][cur_state.l] += 1
           u_dict[cur_state.u]['l'][cur_state.l] += 1
           r_dict[cur_state.r]['l'][cur_state.l] += 1
           d_dict[cur_state.d]['l'][cur_state.l] += 1
       except:
           robot_dict[(cur_state.xr,cur_state.yr)]['l'][cur_state.l] = 1
           game_dict[(cur_state.xg,cur_state.yg)]['l'][cur_state.l] = 1
           u_dict[cur_state.u]['l'][cur_state.l] = 1
           r_dict[cur_state.r]['l'][cur_state.l] = 1
           d_dict[cur_state.d]['l'][cur_state.l] = 1

       #Detect game dynamics
       if cur_state.xg > last_state.xg: #moved right
           try:
               robot_dict[(cur_state.xr,cur_state.yr)]['game']['right'] += 1
               u_dict[cur_state.u]['game']['right'] += 1
               r_dict[cur_state.r]['game']['right'] += 1
               d_dict[cur_state.d]['game']['right'] += 1
               l_dict[cur_state.l]['game']['right'] += 1
           except:
               robot_dict[(cur_state.xr,cur_state.yr)]['game']['right'] = 1
               u_dict[cur_state.u]['game']['right'] = 1
               r_dict[cur_state.r]['game']['right'] = 1
               d_dict[cur_state.d]['game']['right'] = 1
               l_dict[cur_state.l]['game']['right'] = 1

       elif cur_state.xg < last_state.xg: #moved left
           try:
               robot_dict[(cur_state.xr,cur_state.yr)]['game']['left'] += 1
               u_dict[cur_state.u]['game']['left'] += 1
               r_dict[cur_state.r]['game']['left'] += 1
               d_dict[cur_state.d]['game']['left'] += 1
               l_dict[cur_state.l]['game']['left'] += 1
           except:
               robot_dict[(cur_state.xr,cur_state.yr)]['game']['left'] = 1
               u_dict[cur_state.u]['game']['left'] = 1
               r_dict[cur_state.r]['game']['left'] = 1
               d_dict[cur_state.d]['game']['left'] = 1
               l_dict[cur_state.l]['game']['left'] = 1
        
       elif cur_state.yg < last_state.yg: #moved down
           try:
               robot_dict[(cur_state.xr,cur_state.yr)]['game']['down'] += 1
               u_dict[cur_state.u]['game']['down'] += 1
               r_dict[cur_state.r]['game']['down'] += 1
               d_dict[cur_state.d]['game']['down'] += 1
               l_dict[cur_state.l]['game']['down'] += 1
           except:
               robot_dict[(cur_state.xr,cur_state.yr)]['game']['down'] = 1
               u_dict[cur_state.u]['game']['down'] = 1
               r_dict[cur_state.r]['game']['down'] = 1
               d_dict[cur_state.d]['game']['down'] = 1
               l_dict[cur_state.l]['game']['down'] = 1

       elif cur_state.yg > last_state.yg: #moved up
           try:
               robot_dict[(cur_state.xr,cur_state.yr)]['game']['up'] += 1
               u_dict[cur_state.u]['game']['up'] += 1
               r_dict[cur_state.r]['game']['up'] += 1
               d_dict[cur_state.d]['game']['up'] += 1
               l_dict[cur_state.l]['game']['up'] += 1
           except:
               robot_dict[(cur_state.xr,cur_state.yr)]['game']['up'] = 1
               u_dict[cur_state.u]['game']['up'] = 1
               r_dict[cur_state.r]['game']['up'] = 1
               d_dict[cur_state.d]['game']['up'] = 1
               l_dict[cur_state.l]['game']['up'] = 1
       elif cur_state.xg == last_state.xg and cur_state.yg == last_state.yg: #no movement
           try:
               robot_dict[(cur_state.xr,cur_state.yr)]['game']['stop'] += 1
               u_dict[cur_state.u]['game']['stop'] += 1
               r_dict[cur_state.r]['game']['stop'] += 1
               d_dict[cur_state.d]['game']['stop'] += 1
               l_dict[cur_state.l]['game']['stop'] += 1
           except:
               robot_dict[(cur_state.xr,cur_state.yr)]['game']['stop'] = 1
               u_dict[cur_state.u]['game']['stop'] = 1
               r_dict[cur_state.r]['game']['stop'] = 1
               d_dict[cur_state.d]['game']['stop'] = 1
               l_dict[cur_state.l]['game']['stop'] = 1
 

print(robot_dict[(2,4)]['u'])
print(robot_dict[(2,4)]['d'])
print(robot_dict[(1,1)]['u'])
print("=====")
print(game_dict[(2,4)]['u'])
print(game_dict[(2,4)]['d'])
print(game_dict[(1,1)]['u'])
print("=====")
print(robot_dict[(2,4)]['game'])
print(robot_dict[(3,3)]['game'])
print(robot_dict[(1,3)]['game'])
print(robot_dict[(2,2)]['game'])
print(robot_dict[(3,2)]['game'])
print("=====")
print(u_dict[True]['game'])
print(u_dict[False]['game'])
print(l_dict[True]['game'])
print(l_dict[False]['game'])
print("=====")
print(u_dict[True]['d'])
print(u_dict[False]['l'])
print(l_dict[True]['u'])
print(l_dict[False]['r'])

#Code for discovering structures
objects = ["robot","u","d","l","r","game"]
object_dynamics ={"u": ["u_press"], "d": ["d_press"], "l": ["l_press"], "r":["r_press"], "game":["up","right","down","left","stop"]}

dynamic_object1 = {"u_press": "robot", "d_press": "robot", "l_press": "robot", "r_press": "robot", "up" : "u", "right": "r", "down": "d", "left": "l"}
#figure out how to generate the above
#figure out how to get how likely above chain is
#demonstrate how to convert this into transition function (Hint, it's really easy, just normalize the above probabilities and sample from them when applying update dynamics according to object_dynamics)
