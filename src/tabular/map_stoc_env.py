import numpy as np

from tabular import map_env as me

class MapStocEnv(me.MapEnv):

    def __init__(self,map_name,start_r,start_c,maps=None):

        me.MapEnv.__init__(self,map_name,maps,start_r,start_c)
        self.main_count = 0
        self.rand_count = 0

        
    def step(self,action):

        state = self._agent._current_state
        self._agent.move(action)

        main_action_prob = 0.5
        rand_num = np.random.uniform(0,1)
        
        if rand_num < main_action_prob:
            self.main_count += 1
            return self._trans[state][action]
        else:
            action = np.random.choice(self._agent._action_space_n)
            self.rand_count += 1
            return self._trans[state][action]
