import random
import numpy
import sys
from collections import defaultdict
from simple_rl.planning.PlannerClass import Planner
import sys
if sys.version_info[0] < 3:
	import Queue as queue
else:
	import queue

class Dyna(Planner):
    def __init__(self, mdp, name="dyna", max_iterations=500):
        Planner.__init__(self, mdp, name=name)
        self.max_iterations = max_iterations
        self.value_func = defaultdict(float)
        self.max_q_act_histories = defaultdict(str)
        self.reachability_done = False
        self.has_computed_matrix = False
        self.bellman_backups = 0
        self.epsilon = 0.5 #epsilon-decay should be implemented
        self.trans_dict = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: 0)))
        self.reward_dict = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: defaultdict(lambda: 0))))
        self.trans_prob = defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
        self.reward_prob = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: defaultdict(float))))
        self.default_q = 0
        self.alpha=0.1 #step-size
        #initialize all Q(s,a) as zero
        self.q_func = defaultdict(lambda: defaultdict(lambda: self.default_q))
        self.N = 10
        self.previous_record = {} #keep track of previously visited s-a pairs

    def run_dyna(self):
        num_actions = 0  
        #loop
        iteration = 0
        while iteration < self.max_iterations:
            cur_state = self.init_state

            while not cur_state.is_terminal():
                num_actions += 1
                print(num_actions)
                #choose action a by epsilon-greedy policy Q(s,)
                a = self.epsilon_greedy_q_policy(cur_state)

                #execute action, and observe s' and r
                next_state = self.transition_func(cur_state, a)
                reward = self.reward_func(cur_state, a, next_state)

                #Q-learning update Q(s,a) <- Q(s,a) + alpha [r+ gamma * max Q(s',a') - Q(s,a)]
                self.q_func[cur_state][a] +=  self.alpha * (reward + self.mdp.gamma * self.get_max_q_value(next_state) \
                                                        - self.q_func[cur_state][a])

                #update T and R
                #T(s,a,s') --> prob, R(s,a,s',r) --> prob
                self.trans_dict[cur_state][a][next_state] += 1
                self.reward_dict[cur_state][a][next_state][reward] += 1
                self.update_trans_prob() #helper function that converts dict to probability
                self.update_reward_prob() #helper function that converts dict to probability


                if cur_state not in self.previous_record: #tracks the record of previously observed s,a
                    self.previous_record[cur_state] = {a}
                else:
                    self.previous_record[cur_state].add(a)
                cur_state = next_state


                #repeat N times
                for i in range(self.N):
                    #random previously observed state s
                    random_state = random.choice(list(self.previous_record.keys()))
                    #random action a previously taken in s
                    random_action = random.choice(list(self.previous_record[random_state]))

                    # s', r <-- Model (s,a)
                    new_state = self.trans_model(random_state, random_action)
                    new_reward = self.reward_model(random_state, random_action, new_state)

                    #Q(s,a) <- Q(s,a) + alpha [r+ gamma * max Q(s',a') - Q(s,a)]
                    self.q_func[random_state][random_action] += \
                    self.alpha * (new_reward + self.mdp.gamma * self.get_max_q_value(new_state) -
                              self.q_func[random_state][random_action])

            iteration += 1



    def update_trans_prob(self):
        #s,a total count
        for s in self.trans_dict:
            for a in self.trans_dict[s]:
                count = 0
                for s_ in self.trans_dict[s][a]:
                    count += self.trans_dict[s][a][s_]
                for s_ in self.trans_dict[s][a]:
                    self.trans_prob[s][a][s_] = self.trans_dict[s][a][s_] / count


    def update_reward_prob(self):
        for s in self.reward_dict:
            for a in self.reward_dict[s]:
                for s_ in self.reward_dict[s][a]:
                    count = 0
                    for r in self.reward_dict[s][a][s_]:
                        count += self.reward_dict[s][a][s_][r]
                    for r in self.reward_dict[s][a][s_]:
                        self.reward_prob[s][a][s_][r] = self.reward_dict[s][a][s_][r] / count


    def trans_model(self, s, a):
        rand = random.random()
        c = 0
        for s_ in self.trans_prob[s][a]:
            c += self.trans_prob[s][a][s_]
            if rand <= c:
                return s_


    def reward_model(self, s, a, s_):
        rand = random.random()
        c = 0
        for r in self.reward_prob[s][a][s_]:
            c += self.reward_prob[s][a][s_][r]
            if rand <= c:
                return r


    def epsilon_greedy_q_policy(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): action.
        '''
        # Policy: Epsilon of the time explore, otherwise, greedyQ.
        if numpy.random.random() > self.epsilon:
            # Exploit.
            action = self.get_max_q_action(state)
        else:
            # Explore
            action = numpy.random.choice(self.actions)

        return action



    def _compute_matrix_from_trans_func(self):
        if self.has_computed_matrix:
            self._compute_reachable_state_space()
            # We've already run this, just return.
            return

        for s in self.get_states():
            for a in self.actions:
                for sample in range(self.sample_rate):
                    s_prime = self.transition_func(s, a)
                    self.trans_dict[s][a][s_prime] += 1.0 / self.sample_rate

        self.has_computed_matrix = True

    def get_q_value(self, s, a):
        '''
        Args:
            s (State)
            a (str): action
        Returns:
            (float): The Q estimate given the current value function @self.value_func.
        '''
        # Compute expected value.
        expected_future_val = 0
        for s_prime in self.trans_dict[s][a].keys():
            if not s.is_terminal():
                expected_future_val += self.trans_dict[s][a][s_prime] * self.reward_func(s, a, s_prime) + \
                                       self.gamma * self.trans_dict[s][a][s_prime] * self.value_func[s_prime]
                
        return expected_future_val

    def _compute_reachable_state_space(self):
        '''
        Summary:
            Starting with @self.start_state, determines all reachable states
            and stores them in self.states.
        '''

        if self.reachability_done:
            return

        state_queue = queue.Queue()
        state_queue.put(self.init_state)
        self.states.add(self.init_state)

        while not state_queue.empty():
            s = state_queue.get()
            for a in self.actions:
                for samples in range(self.sample_rate): # Take @sample_rate samples to estimate E[V]
                    next_state = self.transition_func(s,a)

                    if next_state not in self.states:
                        self.states.add(next_state)
                        state_queue.put(next_state)

        self.reachability_done = True

    def get_states(self):
        if self.reachability_done:
            return list(self.states)
        else:
            self._compute_reachable_state_space()
            return list(self.states)

    def _compute_max_qval_action_pair(self, state):
        '''
        Args:
            state (State)

        Returns:
            (tuple) --> (float, str): where the float is the Qval, str is the action.
        '''
        # Grab random initial action in case all equal

        best_action = random.choice(self.actions)
        max_q_val = float("-inf")
        shuffled_action_list = self.actions[:]
        random.shuffle(shuffled_action_list)

        # Find best action (action w/ current max predicted Q value)
        for action in shuffled_action_list:
            q_s_a = self.get_q_value(state, action)
            if q_s_a > max_q_val:
                max_q_val = q_s_a
                best_action = action

        return max_q_val, best_action

    def get_max_q_action(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): denoting the action with the max q value in the given @state.
        '''
        return self._compute_max_qval_action_pair(state)[1]

    def get_max_q_value(self, state):
        '''
        Args:
            state (State)

        Returns:
            (float): denoting the max q value in the given @state.
        '''
        return self._compute_max_qval_action_pair(state)[0]

    def get_value(self, state):
        '''
        Args:
            state (State)

        Returns:
            (float)
        '''
        return self.get_max_q_value(state)

    def get_q_value(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            (float): denoting the q value of the (@state, @action) pair.
        '''
        return self.q_func[state][action]
