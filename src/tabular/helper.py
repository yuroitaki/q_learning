import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np

def playGame(t_agent,maze,game_step,no_play,mode="play"):
    
    goals = []
    
    for episode in range(no_play):        
        state = maze.reset()
        acc_reward = 0
        step_count = 0
            
        while step_count <= game_step:

            if mode == "play":
                action = t_agent.play(state,episode)
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


def evalEpisode(goals,num_episode,episode_window,title):

    x = [i for i in range(episode_window-1,num_episode)]
    fig = plt.figure(figsize=(32,16))

    plt.scatter(x,goals,marker='x',c='r')
    plt.title(title,fontweight='bold')
    plt.xlabel("Episode No.")
    plt.ylabel("Moving Average Score")
    plt.show()
    plt.close()


    
def avgEvalEpisode(mean,interval_size,max_r,min_r,num_episode,episode_window,title,save,folder):

    x = [i for i in range(episode_window-1,num_episode)]
    fig = plt.figure(figsize=(32,16))
    
    plt.errorbar(x,mean,yerr=interval_size,ecolor='c',fmt='b-')
    plt.ylim(min_r,max_r)
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


def confInterval(goal_run,conf_lvl,max_reward):

    goal_len = len(goal_run[0])
    mean_conf_goal = []
    interval_goal = np.zeros((2,goal_len))
    
    for i in range(goal_len):
        series_reward = []
        
        for j in range(len(goal_run)):
            series_reward.append(goal_run[j][i])
            
        mean = np.mean(series_reward)
        sem = st.sem(series_reward)
        
        ######## Standard Error of Mean Method ########
        lower_size = sem
        upper_size = sem

        ######## 95% Confidence Level Method ##########
        
        # if sem == 0:
        #     sem = 1e-10        
        # interval_size = sem * st.t.ppf((1+conf_lvl)/2,goal_len-1)

        # lower_size = interval_size
        # upper_size = interval_size

        ################################################
        
        if(mean + upper_size > max_reward):
            upper_size = max_reward - mean
            
        interval_goal[0][i] = lower_size
        interval_goal[1][i] = upper_size

        mean_conf_goal.append(mean)        
        
    return mean_conf_goal, interval_goal

        
    

    
