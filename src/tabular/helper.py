import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np

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
        

def evalEpisode(mean,lower_bound,upper_bound,num_episode,episode_window,title,save,folder):

    x = [i for i in range(episode_window-1,num_episode)]
    fig = plt.figure(figsize=(32,16))
    plt.scatter(x,mean,marker='x',c='b')
    plt.scatter(x,lower_bound,marker='o',c='r')
    plt.scatter(x,upper_bound,marker='*',c='g')
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


def confInterval(goal_run,conf_lvl):

    goal_len = len(goal_run[0])
    mean_conf_goal = []
    
    for i in range(goal_len):
        series_reward = []
        
        for j in range(len(goal_run)):
            series_reward.append(goal_run[j][i])
            
        mean = np.mean(series_reward)
        sem = st.sem(series_reward)
        upper, lower = st.t.interval(conf_lvl,goal_len-1,loc=mean,scale=sem)

        mean_conf_goal.append((mean,upper,lower))

    return mean_conf_goal

        
    

    
