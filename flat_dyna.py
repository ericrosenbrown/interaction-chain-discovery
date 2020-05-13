import matplotlib as plt
import numpy as np
import time
import argparse
from pdb import set_trace as bp
# Other imports
from FlatMDPClass import FlatMDP
from FlatStateClass import FlatState
from dyna import Dyna 

def generate_MDP(width, height, init_loc, goal_locs, gamma, slip_prob):

    actual_args = {
        "width": width, 
        "height": height, 
        "init_loc": init_loc,
        "goal_locs": goal_locs, 
        "gamma": gamma, 
        "slip_prob": slip_prob,
        "step_cost": 0.1
    }

    return FlatMDP(**actual_args)


def main():
    # This accepts arguments from the command line with flags.
    # Example usage: python value_iteration_demo.py -w 4 -H 3 -s 0.05 -g 0.95 -il [(0,0)] -gl [(4,3)] -ll [(4,2)]  -W [(2,2)]
    parser = argparse.ArgumentParser(description='Run a demo that shows a visualization of value iteration on a GridWorld MDP')

    # Add the relevant arguments to the argparser
    parser.add_argument('-w', '--width', type=int, nargs="?", const=5, default=5,
    help='an integer representing the number of cells for the GridWorld width')
    parser.add_argument('-H', '--height', type=int, nargs="?", const=5, default=5,
    help='an integer representing the number of cells for the GridWorld height')
    parser.add_argument('-s', '--slip', type=float, nargs="?", const=0.05, default=0.05,
    help='a float representing the probability that the agent will "slip" and not take the intended action')
    parser.add_argument('-g', '--gamma', type=float, nargs="?", const=0.95, default=0.95,
    help='a float representing the decay probability for Value Iteration')
    parser.add_argument('-il', '--i_loc', type=tuple, nargs="?", const=(0,0), default=(0,0),
    help='two integers representing the starting cell location of the agent [with zero-indexing]')
    parser.add_argument('-gl', '--g_loc', type=list, nargs="?", const=[(3,3)], default=[(3,3)],
    help='a sequence of integer-valued coordinates where the agent will receive a large reward and enter a terminal state')
    args = parser.parse_args()
    mdp = generate_MDP(
        args.width, 
        args.height,
        args.i_loc,
        args.g_loc,
        args.gamma, 
        args.slip)


    #Run Dyna
    dyna = Dyna(mdp, max_iterations=500)
    dyna.run_dyna() #what should run_dyna return?
    exit() #I just stopped here

    for value_dict in histories:
        print("========================")
        for k in value_dict.keys():
            print(str(k) + " " + str(value_dict[k]))
        print(dyna.plan(state=mdp.init_state))


if __name__ == "__main__":
    main()
