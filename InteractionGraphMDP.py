''' InteractionGraphMDPClass.py: Contains the InteractionGraphMDP class. '''

# Python imports.
from __future__ import print_function
import random
import sys, os
import copy
import numpy as np
from collections import defaultdict

# Other imports.
from simple_rl.mdp.MDPClass import MDP
from FlatStateClass import FlatState

# Fix input to cooperate with python 2 and 3.
try:
   input = raw_input
except NameError:
   pass

class InteractionGraphMDP(MDP):
    ''' Class for a generic Interaction Graph MDP '''

    def __init__(self,ichain,ood,actions,init_state,gamma,goals):
        '''
        Args:
            ichain: (DiGraph) edges represent a. predessecor  object to active dynamic, b. active dynamic to associated object
            ood: (Object: State: Object: Dynamic: Value: [0,1]) a dictionary that has the probability that when a predeccesor object is in some state, a successor object will activate a specific dynamic
            actions: (List) list of actions the robot can perform
            init_state: (InteractionGraphMDPState): initial state
            gamma: (float) discount factor for plannng
            goals: (List) list of states representing goals
        '''

        MDP.__init__(self, actions, self._transition_func, self._reward_func, init_state=init_state, gamma=gamma)

    def is_goal_state(self, state):
        return (state.xg, state.yg) in self.goal_locs

    def _reward_func(self, state, action, next_state):
        '''
        Args:
            state (State)
            action (str)
            next_state (State)

        Returns
            (float)
        '''
        if (int(next_state.xg), int(next_state.yg)) in self.goal_locs:
            return 1.0 - self.step_cost
        else:
            return 0 - self.step_cost


    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''
        if state.is_terminal():
            return state
        
        if  self.slip_prob > random.random():
            # Flip dir.
            if action == "up":
                action = random.choice(["left", "right"])
            elif action == "down":
                action = random.choice(["left", "right"])
            elif action == "left":
                action = random.choice(["up", "down"])
            elif action == "right":
                action = random.choice(["up", "down"])
            elif action == "stop":
                action = "stop"

        #Move robot state
        if action == "up" and state.yr < self.height:
            n_xr = state.xr
            n_yr = state.yr + 1
        elif action == "down" and state.yr > 1:
            n_xr = state.xr
            n_yr = state.yr - 1
        elif action == "right" and state.xr < self.width:
            n_xr = state.xr + 1
            n_yr = state.yr
        elif action == "left" and state.xr > 1:
            n_xr = state.xr - 1
            n_yr = state.yr
        else:
            n_xr = state.xr
            n_yr = state.yr


        #Change button pose
        #These hard coded state values would be learned
        n_u =  n_xr == 2 and n_yr == 4
        n_r =  n_xr == 3 and n_yr == 3
        n_d =  n_xr == 2 and n_yr == 2
        n_l =  n_xr == 1 and n_yr == 3

        #change game avatar pose. No slippage for now.
        #currently, actions override each other
        #Move robot state
        if n_u:# and state.yg < self.height:
            n_xg = state.xg
            n_yg = state.yg + 1
        elif n_d:# and state.yg > 1:
            n_xg = state.xg
            n_yg = state.yg - 1
        elif n_r:# and state.xg < self.width:
            n_xg = state.xg + 1
            n_yg = state.yg
        elif n_l:# and state.xg > 1:
            n_xg = state.xg - 1
            n_yg = state.yg
        else:
            n_xg = state.xg
            n_yg = state.yg

        next_state = FlatState(n_xr,n_yr,n_u,n_r,n_d,n_l,n_xg,n_yg)
       

        landed_in_term_goal = (next_state.xg, next_state.yg) in self.goal_locs and self.is_goal_terminal
        if landed_in_term_goal:
            next_state.set_terminal(True)

        return next_state

    def __str__(self):
        return self.name + "_h-" + str(self.height) + "_w-" + str(self.width)

    def __repr__(self):
        return self.__str__()

    def get_goal_locs(self):
        return self.goal_locs

    def visualize_policy(self, policy):
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.grid_world.grid_visualizer import _draw_state

        action_char_dict = {
            "up":"^",       #u"\u2191",
            "down":"v",     #u"\u2193",
            "left":"<",     #u"\u2190",
            "right":">",    #u"\u2192"
        }

        mdpv.visualize_policy(self, policy, _draw_state, action_char_dict)

    def visualize_agent(self, agent):
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.grid_world.grid_visualizer import _draw_state
        mdpv.visualize_agent(self, agent, _draw_state)

    def visualize_value(self):
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.grid_world.grid_visualizer import _draw_state
        mdpv.visualize_value(self, _draw_state)

    def visualize_learning(self, agent, delay=0.005, num_ep=None, num_steps=None):
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.grid_world.grid_visualizer import _draw_state
        mdpv.visualize_learning(self, agent, _draw_state, delay=delay, num_ep=num_ep, num_steps=num_steps)
        input("Press anything to quit")

    def visualize_interaction(self):
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.grid_world.grid_visualizer import _draw_state
        mdpv.visualize_interaction(self, _draw_state)

def _error_check(state, action):
    '''
    Args:
        state (State)
        action (str)

    Summary:
        Checks to make sure the received state and action are of the right type.
    '''

    if action not in FlatMDP.ACTIONS:
        raise ValueError("(simple_rl) FlatError: the action provided (" + str(action) + ") was invalid in state: " + str(state) + ".")

    if not isinstance(state, FlatState):
        raise ValueError("(simple_rl) FlatError: the given state (" + str(state) + ") was not of the correct class.")


def main():
    grid_world = FlatMDP(5, 10, (1, 1), (6, 7))
    grid_world.visualize_policy()

if __name__ == "__main__":
    main()
