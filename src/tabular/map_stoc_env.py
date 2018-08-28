import numpy as np

from tabular import map_env as me

class MapStocEnv(me.MapEnv):

    def __init__(self,map_name,start_r,start_c,anti_stoc_factor,maps=None,
                 stoc_state=None,stoc_act=None,stoc_tres=None,low_r=None,high_r=None):

        me.MapEnv.__init__(self,map_name,maps,start_r,start_c,
                           stoc_state,stoc_act,stoc_tres,low_r,high_r)
        self.main_count = 0
        self.rand_count = 0
        self.anti_stoc_factor = anti_stoc_factor
        
        
    def step(self,action,game_step):

        state = self._agent._current_state
        self.ori_act_record.append(action)
        
        main_action_prob = self.anti_stoc_factor
        rand_num = np.random.uniform(0,1)

        if len(self.act_record) > game_step + 1:
            self.resetActRecord()
        
        if rand_num < main_action_prob:
            self.main_count += 1
            self.act_record.append(action)
            self._agent.move(action)
            return self._trans[state][action]
        else:
            action = np.random.choice(self._agent._action_space_n)
            self.rand_count += 1
            self.act_record.append(action)
            self._agent.move(action)
            return self._trans[state][action]

