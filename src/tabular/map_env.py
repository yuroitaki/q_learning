import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import cprint

from tabular import map_agent as ag

MAPS = {
    "windy_maze":[
        ["O","X","O","F"],
        ["O","O","O","O"],
        ["O","X","O","X"],
        ["O","O","O","O"],
    ],
    "hard_windy_maze":[
        ["X","X","O","O","O","O","O","F"],
        ["X","X","O","X","X","O","X","O"],
        ["O","X","O","O","X","O","O","O"],
        ["O","O","O","O","O","O","O","X"],
        ["O","X","X","X","X","O","O","X"],
        ["O","O","O","O","X","O","X","O"],
        ["O","O","X","O","O","O","O","O"],
        ["O","O","O","O","O","X","O","X"],
    ],
    "risky_windy_maze":[
        ["S","O","O","O","O","O","O","F"],
        ["X","X","O","X","X","O","X","O"],
        ["O","X","O","O","X","O","O","O"],
        ["O","O","O","O","O","O","O","X"],
        ["O","X","X","X","X","O","O","X"],
        ["O","O","O","O","X","O","X","O"],
        ["O","O","X","O","O","O","O","O"],
        ["O","O","O","O","O","X","O","X"],
    ]

}


ACTIONS = {
    # 0: "L",
    # 1: "U",
    # 2: "R",
    # 3: "D"
    0: "←",
    1: "↑",
    2: "→",
    3: "↓"

}


class MapEnv:
    
    def __init__(self,map_name,maps,start_r,start_c):

        if(maps==None):
            maps = MAPS[map_name]
        self.map_name = map_name
        self._maps = np.asarray(maps)
        self._map_length, self._map_width = self._maps.shape
        self._obs_space_n = self._map_length * self._map_width
        self._start_row = start_r
        self._start_col = start_c
        self._start_state = self.toState(self._start_row,self._start_col)
        
        self._agent = ag.MapAgent(self,self._start_state,self._start_row,self._start_col)
        self._trans = {state: {action: [] for action in range(self._agent._action_space_n)} for state in range(self._obs_space_n)}

        self.value_map = np.zeros([self._map_length,self._map_width])
        # self.action_map = {row: {col: "" for col in range(self._map_width)} for row in range(self._map_length)}
        self.action_map = np.empty([self._map_length,self._map_width],dtype="U4")
        
        self.computeTransition()
        
        
    def computeTransition(self):
        
        for row in range(self._map_length):
            for col in range(self._map_width):
               state = self.toState(row,col)
               mark = self._maps[row][col]
               end_game = self.endGame(mark)
               
               for action in range(self._agent._action_space_n):
                   trans = self._trans[state][action]
    
                   if end_game:
                       default_reward = self.reward(None)
                       trans.extend((state,default_reward,end_game))
                        
                   else:
                       new_rol,new_col = self._agent.move(action,row,col)
                       new_state = self._agent._current_state
                       new_mark = self._maps[new_rol][new_col]
                       new_end_game = self.endGame(new_mark)
                       reward = self.reward(new_mark)
                       trans.extend((new_state,reward,new_end_game))


    def step(self,action):

        state  = self._agent._current_state
        self._agent.move(action)
        if self.map_name == "risky_windy_maze":
            if state == 1 and action == 0:
                trans = self._trans[state][action]
                trans[1] = self.reward("S")
                return trans
        return self._trans[state][action]

    
    def randomSampling(self):
        
        action = np.random.random_integers(0,self._agent._action_space_n-1)
        return action

    
    def endGame(self,mark):
        
        if mark in "XFS":
            return True
        else:
            return False
    

    def reward(self,mark):

        if mark == "F":
            return 1
        elif mark == "S":
            treshold_prob = 0.5
            rand_num = np.random.uniform(0,1)
            if rand_num > treshold_prob:
                return 0
            else:
                return 2
        else:
            return 0

        
    def toState(self,row,col):
        
        return (row * self._map_width) + col

    
    def reset(self):

        self._agent.updateState(self._start_row,self._start_col)
        return self._agent._current_state

    
    def setStart(self,row,col):
        
        self._agent.updateState(row,col)
        return self._agent._current_state

    
    def render(self):
        
        for row in range(self._map_length):
            for col in range(self._map_width):
                if(row == self._agent._current_row) and (col == self._agent._current_col):
                    cprint(self._maps[row][col],'white','on_magenta',end="")
                    sys.stdout.write(" ")
                else:
                    sys.stdout.write(self._maps[row][col])
                    sys.stdout.write(" ")
            sys.stdout.write("\n")
        sys.stdout.write("\n")


    def initialiseTable(self,table,val):

        for row in range(self._map_length):
            for col in range(self._map_width):
                state  = self.toState(row,col)
                mark = self._maps[row][col]

                if mark == "O":
                    for action in range(self._agent._action_space_n):
                        table[state][action] = val                          # yields better result, faster to converge
                        # if val > 0.1:
                        #     table[state][action] = val + np.random.uniform(-0.1,0.1)
                        # else:
                        #     table[state][action] = val + np.random.uniform(-0.1,0.1)
                        # table[state][action] = val + np.random.uniform()

            

    def visualiseValFunc(self,val_func,act_choice):

        self.convertValFunc(val_func,act_choice)
        sns.heatmap(self.value_map,annot=self.action_map,xticklabels=False,yticklabels=False,fmt='')
        plt.show()
                

    def convertValFunc(self,val_func,act_choice):
        
        row = 0
        for state in range(self._obs_space_n):
            col = state % self._map_width
            self.value_map[row][col] = val_func[state]
            
            mark = self._maps[row][col]
            if (mark == "O"):
                action = self.convertActionToLetter(act_choice[state])
                self.action_map[row][col] = action
            
            if col == self._map_width - 1:
                row += 1

        
    def convertActionToLetter(self,act_list):

        act_letter = ""
        
        for act in act_list:
            act_letter += ACTIONS[act]

        return act_letter
        
        
        
def makeMapEnv(map_name,start_r=2,start_c=0,maps=None):
        
    maze = MapEnv(map_name,maps,start_r,start_c)
    return maze

    
