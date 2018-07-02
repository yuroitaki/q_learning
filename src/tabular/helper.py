import matplotlib.pyplot as plt

def playGame(t_agent,maze,game_step,mode="play"):
    
    goals = []
    
    for episode in range(t_agent.max_epi):        
        state = maze.reset()
        acc_reward = 0
        step_count = 0
            
        while step_count <= game_step:

            if mode == "play":
                action = t_agent.play(state)
            elif mode == "rand":
                action = maze.randomSampling()
            new_state, reward, done = maze.step(action)
            acc_reward += reward
            state = new_state
            step_count+=1
            
            if done == True:
                break
        goals.append(acc_reward)
        
    return goals
        

def evalEpisode(score,num_episode,episode_window,title,save,folder):
    
    y = calcMovingAverage(score,episode_window)
    x = [i for i in range(episode_window-1,num_episode)]
    fig = plt.figure(figsize=(32,16))
    plt.scatter(x,y,marker='x')
    plt.title(title,fontweight='bold')
    plt.xlabel("Episode No.")
    plt.ylabel("Moving Average Score")
    plt.show()
    if save == True:
        fig.savefig("/vol/bitbucket/ttc14/thesis/fig/tabular/{0}/{1}.png".format(folder,title),dpi=100)
    plt.close()


def calcMovingAverage(score,episode_window):

    start = 0
    end = episode_window-1
    final = len(score)-1
    y = []
    
    while end <= final:
        if(start==0):
            first_mean = sum(score[start:end+1])/episode_window
            y.append(first_mean)
        else:
            mean = y[start-1] - (score[start-1]/episode_window) + (score[end]/episode_window)
            y.append(mean)
        start += 1
        end += 1
        
    return y

    
def writeResult(filename,folder,params,string_param,run):
    
    with open("/vol/bitbucket/ttc14/thesis/result/tabular/{0}/{1}.txt".format(folder,filename),"a+") as f:
        if(run==0):
            f.write(string_param)
            f.write("\n")
        for item in params:
            f.write(str(item))
            f.write(" ")
        f.write("\n")


def storeTable(filename,folder,table,table_param,run):

    with open("/vol/bitbucket/ttc14/thesis/result/tabular/{0}/tables/{1}.txt".format(folder,filename),"a+") as f:
        if(run==0):
            f.write(table_param)
            f.write("\n")
            
        for item in table:
            f.write(str(item))
            f.write("\n")
        f.write("\n\n\n")
