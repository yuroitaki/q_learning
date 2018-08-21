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


def evalAvg(mean,err_up,err_down,max_r,min_r,num_episode,
            episode_window,title,save,folder,fmt_col,label):

    x = [i for i in range(episode_window-1,num_episode)]
    fig = plt.figure(figsize=(32,16))            

    mean_0 = mean[0]
    err_up_0 = err_up[0]
    err_down_0 = err_down[0]
    label_0 = label[0]
    plt.plot(x,mean_0,color=fmt_col,label=label_0)
    plt.fill_between(x,mean_0+err_up_0,mean_0-err_down_0,color=fmt_col,alpha=0.2)
    
    len_mean = len(mean)
    
    if len_mean > 1:
        mean_1 = mean[1]
        err_up_1 = err_up[1]
        err_down_1 = err_down[1]
        label_1 = label[1]
        plt.plot(x,mean_1,color="m",label=label_1,alpha=0.8)
        plt.fill_between(x,mean_1+err_up_1,mean_1-err_down_1,color="m",alpha=0.2)

        if len_mean > 2:
            mean_2 = mean[2]
            err_up_2 = err_up[2]
            err_down_2 = err_down[2]
            label_2 = label[2]
            plt.plot(x,mean_2,color="g",label=label_2,alpha=0.8)
            plt.fill_between(x,mean_2+err_up_2,mean_2-err_down_2,color="g",alpha=0.2)


            if len_mean > 3:
                mean_3 = mean[3]
                err_up_3 = err_up[3]
                err_down_3 = err_down[3]
                label_3 = label[3]
                plt.plot(x,mean_3,color="c",label=label_3,alpha=0.8)
                plt.fill_between(x,mean_3+err_up_3,mean_3-err_down_3,color="c",alpha=0.2)
 
            
    plt.ylim(min_r,max_r)
    plt.title(title,fontweight='bold')
    plt.xlabel("Episode No.")
    plt.ylabel("Moving Average Score")
    plt.legend()
    
    plt.show()
    if save == True:
        fig.savefig("/vol/bitbucket/ttc14/thesis/fig/tabular/{0}/{1}.png".format(folder,title),dpi=100)
    plt.close()

    
def saveGraphData(data,title):

    data_file = "/vol/bitbucket/ttc14/thesis/pickle/{}.pkl".format(title)
    data_open = open(data_file,"wb")
    pickle.dump(data,data_open)
    data_open.close()
    

def readGraphData(title):

    data_file = "/vol/bitbucket/ttc14/thesis/pickle/{}.pkl".format(title)

    data = None
    
    with open(data_file,"rb") as f:
        data = pickle.load(f)

    mean, err_up, err_down = data
    
    return mean, err_up, err_down


        
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
    mean_conf_goal = np.zeros((goal_len,))
    err_up_goal = np.zeros((goal_len,))
    err_down_goal = np.zeros((goal_len,))
    
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
            
        err_down_goal[i] = lower_size
        err_up_goal[i] = upper_size
        mean_conf_goal[i] = mean       
        
    return mean_conf_goal, err_up_goal, err_down_goal


if __name__ == "__main__":

    game = "hard_windy_maze"          # windy_maze   # hard_windy_maze  # risky_windy_maze
    game_type = "deterministic"                        # deterministic  # stochastic
    q_update = "vanilla"                 # vanilla # count # risk
    exp_strategy = "boltzmann"               # "epsilon", "various", "greedy", "boltzmann"
    
    run = 30                                 # number of runs to train the agent
    max_episode = 350
    
    episode_window = 100                 # size of the window for moving average, use factor of 10
    max_reward = 1.0
    max_r = 1.2                           # upper y bound
    min_r = 0.0                           # lower y bound

    fmt_col = "g"
    save = False                            # True to save the picture generated from evalEpisode()
    folder = "hard_windy_maze"              # windy_maze  # hard_windy_maze

    filename = "{}-{}_{}-strat_{}-explore_{}-runs".format(game_type,game,q_update,exp_strategy,run)

    tag_1 = "_init_1"
    tag_2 = "_init_noise"
    # tag_3 = "_init_noise"
    # tag_4 = "_init_noise"
    

    ####### Solo Exploration ##############
    
    label_1 = tag_1[1:]
    label_2 = tag_2[1:]
    # label_3 = tag_3[1:]
    
    title_1 = filename + tag_1
    title_2 = filename + tag_2
    # title_3 = filename + tag_3

    ######## Secondary Exploration ########
    
    exp_strategy_1 = "greedy"               # "epsilon", "softmax", "greedy", "boltzmann"
    exp_strategy_2 = "boltzmann"               # "epsilon", "softmax", "greedy", "boltzmann"
    exp_strategy_3 = "greedy"               # "epsilon", "softmax", "greedy", "boltzmann"

    # label_1 = exp_strategy_1 + tag_1
    # label_2 = exp_strategy_2 + tag_2
    # label_3 = exp_strategy_1 + tag_3
    # label_4 = exp_strategy_2 + tag_4

    filename_1 = "{}-{}_{}-strat_{}-explore_{}-runs".format(game_type,game,q_update,exp_strategy_1,run)
    filename_2 = "{}-{}_{}-strat_{}-explore_{}-runs".format(game_type,game,q_update,exp_strategy_2,run)
    filename_3 = "{}-{}_{}-strat_{}-explore_{}-runs".format(game_type,game,q_update,exp_strategy_1,run)
    filename_4 = "{}-{}_{}-strat_{}-explore_{}-runs".format(game_type,game,q_update,exp_strategy_2,run)
    
    # title_1 = filename_1 + tag_1
    # title_2 = filename_2 + tag_2
    # title_3 = filename_3 + tag_3
    # title_4 = filename_4 + tag_4

    ################################################
    
    mean_1, err_up_1, err_down_1 = readGraphData(title_1)
    mean_2, err_up_2, err_down_2 = readGraphData(title_2)
    # mean_3, err_up_3, err_down_3 = readGraphData(title_3)
    # mean_4, err_up_4, err_down_4 = readGraphData(title_4)
    
    mean = [mean_1,mean_2]
    err_up = [err_up_1,err_up_2]
    err_down = [err_down_1,err_down_2]
    label = [label_1,label_2]


    # mean = [mean_1,mean_2,mean_3]
    # err_up = [err_up_1,err_up_2,err_up_3]
    # err_down = [err_down_1,err_down_2,err_down_3]
    # label = [label_1,label_2,label_3]

    
    # mean = [mean_1,mean_2,mean_3,mean_4]
    # err_up = [err_up_1,err_up_2,err_up_3,err_up_4]
    # err_down = [err_down_1,err_down_2,err_down_3,err_down_4]
    # label = [label_1,label_2,label_3,label_4]

    # mean = [mean_1]
    # err_up = [err_up_1]
    # err_down = [err_down_1]
    # label = [label_1]
    
    filename += tag_1
    filename += tag_2
    
    evalAvg(mean,err_up,err_down,max_r,min_r,
            max_episode,episode_window,filename,save,
            folder,fmt_col,label)

