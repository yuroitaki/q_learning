import numpy as np

MAPS = {
    "windy_maze":[
        ["O","X","O","F"],
        ["O","O","O","O"],
        ["S","X","O","X"],
        ["O","O","O","O"],
    ]
}


class MapEnv:
    
    def __init__(self,maps,map_name):

        self._start_row = 2
        self._start_col = 0
        
        if(maps==None):
            maps = MAPS[map_name]
        self._maps = np.asarray(maps)
        self._map_length, self._map_width = self._maps.shape
        self._obs_space_n = self._map_length * self._map_width
        self._start_state = self.toState(self._start_row,self._start_col)
        
        self._agent = MapAgent(self,self._start_state,self._start_row,self._start_col)
        

    def computeTransition(self):
        pass

        
    def toState(self,row,col):
        
        return (row * self._map_width) + col


    def reset(self):

        self._agent.updateState(self._start_row,self._start_col)
        

class MapAgent:

    def __init__(self,env,start_state,start_row,start_col):

        self._env = env
        self._action_space_n = 4
        self._current_row = start_row
        self._current_col = start_col
        self._current_state = start_state

        print("An agent is created at starting point",self._current_state)


    def move(self,action):

        if action == 0:          # LEFT
            col = max(self._current_col-1,0)
            row = self._current_row
        elif action == 1:        # UP
            row = max(self._current_row-1,0)
            col = self._current_col
        elif action  == 2:       # RIGHT
            col = min(self._current_col+1,self._env._map_width)
            row = self._current_row
        elif action == 3:        # DOWN
            row = min(self._current_row+1,self._env._map_length)
            col = self._current_col
        else:
            raise KeyError("The action key {0} provided is not valid.".format(action))

        self.updateState(row,col)
        return (row,col)

    
    def updateState(self,row,col):

        self._current_row = row
        self._current_col = col
        self._current_state = self._env.toState(row,col)



        
def makeMapEnv(maps,map_name):
        
    maze = MapEnv(maps,map_name)
    return maze

    

def main():

    maze = makeMapEnv(None,"windy_maze")
    print(maze._agent.move(1))
    print(maze._agent._current_state)
    print(maze._agent.move(0))
    print(maze._agent._current_state)
    maze.reset()
    print(maze._agent._current_state)

if __name__ == "__main__":
    main()
