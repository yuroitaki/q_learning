import numpy as np
import sys
from termcolor import cprint

from tabular import map_agent as ag

MAPS = {
    "windy_maze":[
        ["O","X","O","F"],
        ["O","O","O","O"],
        ["S","X","O","X"],
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
        ["S","O","O","O","O","X","O","X"],
    ]
}


class MapEnv:
    
    def __init__(self,map_name,maps,start_r,start_c):

        if(maps==None):
            maps = MAPS[map_name]
        self._maps = np.asarray(maps)
        self._map_length, self._map_width = self._maps.shape
        self._obs_space_n = self._map_length * self._map_width
        self._start_row = start_r
        self._start_col = start_c
        self._start_state = self.toState(self._start_row,self._start_col)
        
        self._agent = ag.MapAgent(self,self._start_state,self._start_row,self._start_col)
        self._trans = {state: {action: [] for action in range(self._agent._action_space_n)} for state in range(self._obs_space_n)}                
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
        
        return self._trans[state][action]

    
    def randomSampling(self):
        
        action = np.random.random_integers(0,self._agent._action_space_n-1)
        return action

    
    def endGame(self,mark):
        
        if mark in "XF":
            return True
        else:
            return False
    

    def reward(self,mark):

        if mark == "F":
            return 1
        else:
            return 0

        
    def toState(self,row,col):
        
        return (row * self._map_width) + col


    def reset(self):

        self._agent.updateState(self._start_row,self._start_col)
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

        
def makeMapEnv(map_name,maps=None,start_r=2,start_c=0):
        
    maze = MapEnv(map_name,maps,start_r,start_c)
    return maze

    
