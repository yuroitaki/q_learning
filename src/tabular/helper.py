import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import pickle

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


def monteCarlo(t_agent,maze,game_step,no_play,discount):
        
    for row in range(maze._map_length):
        for col in range(maze._map_width):
            mark = maze._maps[row][col]
            if mark == "O":
                for chosen_action in range(t_agent.act_n):
                    sample_goals = []
                    start_state = maze.toState(row,col)
                    
                    for episode in range(no_play):
                        state = maze.setStart(row,col)
                        step_count = 0
                        true_goal = 0
                    
                        while step_count <= game_step:

                            if step_count == 0:
                                action = chosen_action
                            else:
                                action = t_agent.play(state,episode)

                            new_state, reward, done = maze.step(action)

                            true_goal += reward*(discount**step_count)
                            state = new_state
                            step_count += 1

                            if done == True:
                                break
                    
                        sample_goals.append(true_goal)
                    # print("state {}, action {}, goals {}".format(start_state,chosen_action,sample_goals))
                    t_agent.monte_goal[start_state][chosen_action] = np.mean(sample_goals)
                    t_agent.monte_var[start_state][chosen_action] = np.var(sample_goals)


def monteDiff(monte,estimate):

    delta = (monte - estimate)**2
    expected_delta = delta.mean()
    
    return delta,expected_delta
    

def evalMonteDiff(delta,num_epi,epi_window,title):

    x = [i for i in range(0,num_epi,epi_window)]
    fig = plt.figure(figsize=(32,16))

    plt.plot(x,delta)
    plt.title(title,fontweight='bold')
    plt.xlabel("Episode No.")
    plt.ylabel("Delta btwn Monte Carlo and Estimate")
    plt.show()
    plt.close()



def evalEpisode(goals,num_episode,episode_window,title):

    x = [i for i in range(episode_window-1,num_episode)]
    fig = plt.figure(figsize=(32,16))

    plt.scatter(x,goals,marker='x',c='r')
    plt.title(title,fontweight='bold')
    plt.xlabel("Episode No.")
    plt.ylabel("Moving Average Score")
    plt.show()
    plt.close()


    
def avgEvalEpisode(mean,interval_size,max_r,min_r,num_episode,
                   episode_window,title,save,folder,err_col,fmt_col,
                   label_1=None,mean_2=None,err_2=None,label_2=None,
                   mean_3=None,err_3=None,label_3=None):

    x = [i for i in range(episode_window-1,num_episode)]
    # fig = plt.figure(figsize=(32,16))            
    fig, ax = plt.subplots(figsize=(32,16))
    
    mark_1, cap_1, bar_1 = ax.errorbar(x,mean,yerr=interval_size,ecolor=err_col,fmt=fmt_col,label=label_1,ms=10)
    for bar in bar_1:
        bar.set_alpha(0.1)
    if mean_2 is not None and err_2 is not None:
        mark_2, cap_2, bar_2 =ax.errorbar(x,mean_2,yerr=err_2,ecolor="pink",fmt="r-",label=label_2,ms=10)
        [bar.set_alpha(0.1) for bar in bar_2]
        if mean_3 is not None and err_3 is not None:
            mark_3, cap_3, bar_3 =ax.errorbar(x,mean_3,yerr=err_3,ecolor="yellow",fmt="g-",label=label_3,ms=10)
            # for bar in bar_3:
            bar_3.set_alpha(0.1) 
    
    plt.ylim(min_r,max_r)
    plt.title(title,fontweight='bold')
    plt.xlabel("Episode No.")
    plt.ylabel("Moving Average Score")
    plt.legend()
    
    plt.show()
    if save == True:
        fig.savefig("/vol/bitbucket/ttc14/thesis/fig/tabular/{0}/{1}.png".format(folder,title),dpi=100)
    plt.close()


def saveGraphPickle(mean,err,mean_title,err_title):

    mean_file = "/vol/bitbucket/ttc14/thesis/pickle/{}.pkl".format(mean_title)
    err_file = "/vol/bitbucket/ttc14/thesis/pickle/{}.pkl".format(err_title)
    mean_open = open(mean_file,"wb")
    err_open = open(err_file, "wb")
    pickle.dump(mean,mean_open)
    pickle.dump(err,err_open)
    mean_open.close()
    err_open.close()


def readGraphPickle(mean_title,err_title):

    mean_file = "/vol/bitbucket/ttc14/thesis/pickle/{}.pkl".format(mean_title)
    err_file = "/vol/bitbucket/ttc14/thesis/pickle/{}.pkl".format(err_title)

    mean = None
    err = None
    
    with open(mean_file,"rb") as f_mean:
        mean = pickle.load(f_mean)
        
    with open(err_file,"rb") as err_mean:
        err = pickle.load(err_mean)

    return mean, err

        
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


if __name__ == "__main__":

    game = "hard_windy_maze"          # windy_maze   # hard_windy_maze  # risky_windy_maze
    game_type = "deterministic"                        # deterministic  # stochastic
    q_update = "vanilla"                     # vanilla # count # risk
    exp_strategy = "epsilon"               # "epsilon", "softmax", "greedy", "boltzmann"
    run = 30                                 # number of runs to train the agent
    max_episode = 50000
    
    episode_window = 10000                     # size of the window for moving average, use factor of 10
    max_reward = 1.0
    max_r = 1.2                           # upper y bound
    min_r = 0.0                           # lower y bound
    
    save = False                            # True to save the picture generated from evalEpisode()
    folder = "hard_windy_maze"              # windy_maze  # hard_windy_maze
    err_col = "c"
    fmt_col = "b-"
    
    filename = "{}-{}_{}-strat_{}-explore_{}-runs".format(game_type,game,q_update,exp_strategy,run)

    label_1 = "constant"
    label_2 = "linear"
    label_3 = "exponential"
    
    mean_title_1 = filename + "_mean" + "_const_epsilon"
    err_title_1 = filename  + "_err" + "_const_epsilon" 

    mean_title_2 = filename + "_mean" + "_lin_epsilon"
    err_title_2 = filename  + "_err" + "_lin_epsilon" 

    mean_title_3 = filename + "_mean" + "_exp_epsilon"
    err_title_3 = filename  + "_err" + "_exp_epsilon" 

    mean_1, err_1 = readGraphPickle(mean_title_1,err_title_1)
    mean_2, err_2 = readGraphPickle(mean_title_2,err_title_2)
    mean_3, err_3 = readGraphPickle(mean_title_3,err_title_3)
    
    avgEvalEpisode(mean_1,err_1,max_r,min_r,
                   max_episode,episode_window,filename,save,folder,err_col,fmt_col,
                   label_1,mean_2,err_2,label_2,mean_3,err_3,label_3)

