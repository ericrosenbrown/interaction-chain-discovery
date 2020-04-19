from simple_rl.agents import QLearningAgent, RandomAgent, RMaxAgent
from FlatMDPClass import FlatMDP
from simple_rl.run_experiments import run_agents_on_mdp
import random
import copy
#from graph import Graph
import networkx as nx
import matplotlib.pyplot as plt
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
        #if mdp.is_goal_state(mdp.cur_state):
        #    mdp.reset()
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
           if cur_state.l == False:
               print("LOL")
               print(i)
               print(len(episode))
               print(last_state)
               print(cur_state)
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
objs = ["robot","u","d","l","r","game"]
obj_dyn ={"u": ["u_press"], "d": ["d_press"], "l": ["l_press"], "r":["r_press"], "game":["up","right","down","left"]}

true_dyn_succ = {"u_press": "robot", "d_press": "robot", "l_press": "robot", "r_press": "robot", "up" : "u", "right": "r", "down": "d", "left": "l"}

def random_dynamic_successor(obj_dyn):
    ret_graph = {}
    graph_objects = list(obj_dyn.keys())
    for obj in graph_objects:
        potential_successors = copy.deepcopy(graph_objects)
        potential_successors.remove(obj)
        potential_successors.append("robot")
        for dyn in obj_dyn[obj]:
            ret_graph[dyn] = random.choice(potential_successors)
    return(ret_graph)

print(random_dynamic_successor(obj_dyn))

#turn random_graph into a graph based on objct dynamics, if it is a DAG, valid. Otherwise, throwout

def create_ichain(dyn_succ, obj_dyn):
    #create a graph based on objects, dynamics, and succesor relationships, and check if it's valid
    obj_nodes = list(obj_dyn.keys()) + ["robot"]
    dyn_nodes = list(dyn_succ.keys())
    nodes = obj_nodes + dyn_nodes
    num_nodes = len(nodes)

    #generate the graph and add edges according to succesor relationship and obj_dynk
    #g = Graph(num_nodes)
    G = nx.DiGraph()

    #add directed edges from object dynamic to associated object (obj_dyn) (except for robot)
    for obj in obj_dyn.keys():
        for dyn in obj_dyn[obj]:
            G.add_edge(dyn,obj)
  
    #add directed edges from successor objects to dynamics (dyn_succ)
    for dyn in dyn_succ.keys():
        G.add_edge(dyn_succ[dyn],dyn)

    return(G)

def viz_ichain(G):
    #draw graph for sanity
    positions = nx.spring_layout(G)

    nx.draw_networkx_labels(G,positions)
    nx.draw_networkx_edges(G,positions)
    nx.draw_networkx_nodes(G,positions)
    plt.show()

    

true_ichain = create_ichain(true_dyn_succ,obj_dyn)
print(nx.is_directed_acyclic_graph(true_ichain))
#viz_ichain(true_ichain)

looking = True
while looking:
    random_dyn_succ = random_dynamic_successor(obj_dyn)
    random_ichain = create_ichain(random_dyn_succ,obj_dyn)
    if nx.is_directed_acyclic_graph(random_ichain):
        looking = False

viz_ichain(random_ichain)

###compute how likely this graph is to exist

def dict_to_prob(d):
    #normalize values in object-object-dynamic dict
    ret_d = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for pos,pod in d.items(): #for each position robot can take on
        for obj,objd in pod.items(): #for each potential successor object
            total_counts = 0
            for dyn,counts in objd.items(): #for each dynamic successor objecy can take on
                total_counts += counts
            for dyn,counts in objd.items(): #now populate p_robot_dict with probabilities
                ret_d[pos][obj][dyn] = float(counts)/total_counts
    return(ret_d)

p_robot_dict = dict_to_prob(robot_dict)
p_u_dict = dict_to_prob(u_dict)
p_d_dict = dict_to_prob(d_dict)
p_l_dict = dict_to_prob(l_dict)
p_r_dict = dict_to_prob(r_dict)
p_game_dict = dict_to_prob(game_dict)
        
print(p_robot_dict[(2,4)]['game'])
print(p_u_dict[True]['d'])
print(p_u_dict[False]['l'])

