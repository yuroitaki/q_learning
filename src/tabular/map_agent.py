import numpy as np

class MapAgent:

    def __init__(self,env,start_state,start_row,start_col):

        self._action_space_n = 4
        self._env = env
        self._current_row = start_row
        self._current_col = start_col
        self._current_state = start_state

        print("An agent is created at starting point",self._current_state)


    def move(self,action,row=None,col=None):

        if row == None:
            row = self._current_row
        if col == None:
            col = self._current_col 
            
        if action == 0:                             # LEFT
            col = max(col-1,0)
        elif action == 1:                           # UP
            row = max(row-1,0)
        elif action  == 2:                          # RIGHT
            col = min(col+1,self._env._map_width-1)
        elif action == 3:                           # DOWN
            row = min(row+1,self._env._map_length-1)
        else:
            raise KeyError("The action key {0} provided is not valid.".format(action))

        self.updateState(row,col)
        return row, col

    
    def updateState(self,row,col):

        self._current_row = row
        self._current_col = col
        self._current_state = self._env.toState(row,col)
