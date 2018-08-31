import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import pickle

def playGame(t_agent,maze,game_step,no_play,epi,max_epi,mode="play"):
    
    goals = []
    
    for episode in range(no_play):        
        state = maze.reset()
        maze.resetActRecord() 
        acc_reward = 0
        step_count = 0
    
        while step_count <= game_step:

            if mode == "play":
                action = t_agent.play(state,episode)
            elif mode == "rand":
                action = maze.randomSampling()
                
            new_state, reward, done = maze.step(action,game_step)
            acc_reward += reward
            state = new_state
            step_count+=1
            
            if done == True:
                break
        goals.append(acc_reward)
        
    return goals


def plotMap(t_agent,maze,plot_table,mode,title,epi):

    if plot_table == "Q":
        table = t_agent.Q
    elif plot_table == "U":
        table = t_agent.U
    elif plot_table == "var":
        table = t_agent.var
    elif plot_table == "monte_Q":
        table = t_agent.monte_goal
    elif plot_table == "monte_var":
        table = t_agent.monte_var

        
    if mode == "val_act" or mode == "val_func":
        title += "_epi-{}th_optimal-value-{}".format(epi,plot_table)
        t_agent.extractValue(table)
        maze.visualiseValFunc(t_agent.value_func,t_agent.action_choice,mode,title)

    elif mode == "act_rec":
        title += "_epi-{}th_game-play".format(epi,plot_table)
        t_agent.extractValue(table)
        act_arr = maze.insertRealAct()
        maze.visualiseValFunc(t_agent.value_func,act_arr,mode,title)
        
    elif mode == "act":
        
        act_0 = "left"
        act_1 = "up"
        act_2 = "right"
        act_3 = "down"
        
        acts = [act_0,act_1,act_2,act_3]
        ori_title = title
        
        for act in acts:
            title = ori_title
            title += "_{}-epi_act-{}-action-value-{}".format(epi,act,plot_table)
            maze.visualiseValFunc(table,None,act,title)
        
        
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

                            new_state, reward, done = maze.step(action,game_step)

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

    delta = np.abs(monte[:,:-1] - estimate[:,:-1])
    expected_delta = delta.mean()
    return delta, expected_delta
    

def meanMonte(monte,estimate):

    mean_monte = monte[:,:-1].mean()
    mean_est = estimate[:,:-1].mean()

    return mean_monte, mean_est


def evalMonte(est,monte,num_epi,epi_window,title,labels,delta):

    x = [i for i in range(0,num_epi,epi_window)]
    fig = plt.figure(figsize=(32,16))

    plt.plot(x,est,color="r",label="Estimated {}".format(labels))
    plt.plot(x,monte,color="b",label="Monte {}".format(labels))
    plt.plot(x,delta,color="g",label="Delta {}".format(labels))
    plt.title(title,fontweight='bold')
    plt.xlabel("Episode No.")
    plt.ylabel("{} Mean Values".format(labels))
    plt.legend()
    plt.show()
    plt.close()



def evalEpisode(goals,num_episode,episode_window,title):

    x = [i for i in range(episode_window-1,num_episode)]
    fig = plt.figure(figsize=(32,16))

    plt.plot(x,goals,color='r')
    plt.title(title,fontweight='bold')
    plt.xlabel("Episode No.")
    plt.ylabel("Moving Average Score")
    plt.show()
    plt.close()


def plotDeltaRun(mov_est,mov_monte,mov_delta,max_lim,min_lim,
                 max_episode,monte_test_freq,monte_title,save,
                 folder,fmt_col,labels):
    
    mean_data = [mov_est[0],mov_monte[0],mov_delta[0]]
    err_up_data = [mov_est[1],mov_monte[1],mov_delta[1]]
    err_down_data = [mov_est[2],mov_monte[2],mov_delta[2]]
    
    label_est ="Estimated {}".format(labels)
    label_monte ="Monte {}".format(labels)
    label_delta ="Delta {}".format(labels)
    label_data = [label_est,label_monte,label_delta]

    ylabels = "{} Mean Values".format(labels)
    
    evalAvg(mean_data,err_up_data,err_down_data,max_lim,min_lim,
            max_episode,monte_test_freq,monte_title,save,folder,
            fmt_col,label_data,True,ylabels)
    

    
def evalAvg(mean,err_up,err_down,max_r,min_r,num_episode,
            episode_window,title,save,folder,fmt_col,label,delta=False,
            ylabels="Moving Average Score"):

    if delta == False:
        x = [i for i in range(episode_window-1,num_episode)]
    else:
        x = [i for i in range(0,num_episode,episode_window)]
        
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
        plt.plot(x,mean_1,color="r",label=label_1,alpha=0.8)
        plt.fill_between(x,mean_1+err_up_1,mean_1-err_down_1,color="r",alpha=0.2)

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
                plt.plot(x,mean_3,color="m",label=label_3,alpha=0.8)
                plt.fill_between(x,mean_3+err_up_3,mean_3-err_down_3,color="m",alpha=0.2)


                if len_mean > 4:
                    mean_4 = mean[4]
                    err_up_4 = err_up[4]
                    err_down_4 = err_down[4]
                    label_4 = label[4]
                    plt.plot(x,mean_4,color="m",label=label_4,alpha=0.8)
                    plt.fill_between(x,mean_4+err_up_4,mean_4-err_down_4,color="m",alpha=0.2)

        
    plt.ylim(min_r,max_r)
    plt.title(title,fontweight='bold')
    plt.xlabel("Episode No.")
    plt.ylabel(ylabels)
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

    game = "risky_windy_maze"          # windy_maze   # hard_windy_maze  # risky_windy_maze
    game_type = "deterministic"                        # deterministic  # stochastic
    q_update = "risk"                 # vanilla # count # risk
    exp_strategy = "greedy"               # "epsilon", "various", "greedy", "boltzmann"
    
    q_update_1 = "risk"
    # q_update_2 = "vanilla"
    # q_update_3 = "count"
    
    exp_strategy_1 = "greedy"               # "epsilon", "various", "greedy", "boltzmann"
    # exp_strategy_2 = "epsilon"
    # exp_strategy_3 = "boltzmann" 
    
    run = 30                                 # number of runs to train the agent
    max_episode = 5000
    
    episode_window = 500                 # size of the window for moving average, use factor of 10
    max_reward = 4.0
    max_r = 4.2                           # upper y bound
    min_r = 0.0                           # lower y bound

    fmt_col = "b"
    save = False                            # True to save the picture generated from evalEpisode()
    folder = "hard_windy_maze"              # windy_maze  # hard_windy_maze

    filename = "{}-{}_{}-strat_{}-explore_{}-runs".format(game_type,game,q_update,exp_strategy,run)
    
    filename_1 = "{}-{}_{}-strat_{}-explore_{}-runs".format(game_type,game,q_update_1,exp_strategy_1,run)
    filename_2 = "{}-{}_{}-strat_{}-explore_{}-runs".format(game_type,game,q_update_1,exp_strategy_1,run)
    filename_3 = "{}-{}_{}-strat_{}-explore_{}-runs".format(game_type,game,q_update_1,exp_strategy_1,run)
    filename_4 = "{}-{}_{}-strat_{}-explore_{}-runs".format(game_type,game,q_update_1,exp_strategy_1,run)
    # filename_5 = "{}-{}_{}-strat_{}-explore_{}-runs".format(game_type,game,q_update_1,exp_strategy_1,run)

    

    tag_1 = "_vanilla-risk-point25*"
    tag_2 = "_vanilla-risk-1*"
    tag_3 = "_vanilla-risk-2point5*"
    tag_4 = "_vanilla-risk-10*"

    # tag_2 = "_greedy-init-4*"
    # tag_3 = "_epsilon-init-4*"
    # tag_4 = "_boltzmann-init-4*"
    # tag_3 = "_vanilla-risk-100"
    

    ####### Solo Exploration ##############
    
    # label_1 = q_update_1 + "_vanilla"
    # label_2 = exp_strategy_1+ "_init_1"
    # label_3 = q_update_3 + "_const_epsilon"
    # label_4 = tag_4[1:]
    # label_5 = exp_strategy_1 + "_init_1"

    label_1 = tag_1[1:]
    label_2 = tag_2[1:]
    label_3 = tag_3[1:]
    label_4 = tag_4[1:]
    # label_5 = tag_5[1:] 
    
    
    title_1 = filename_1 + tag_1
    title_2 = filename_2 + tag_2
    title_3 = filename_3 + tag_3
    title_4 = filename_4 + tag_4
    # title_5 = filename_5 + tag_5

    ######## Secondary Exploration ########
    
    # exp_strategy_1 = "greedy"               # "epsilon", "softmax", "greedy", "boltzmann"
    # exp_strategy_2 = "boltzmann"               # "epsilon", "softmax", "greedy", "boltzmann"
    # exp_strategy_3 = "greedy"               # "epsilon", "softmax", "greedy", "boltzmann"

    # label_1 = exp_strategy_1 + tag_1
    # label_2 = exp_strategy_2 + tag_2
    # label_3 = exp_strategy_1 + tag_3
    # label_4 = exp_strategy_2 + tag_4

    # filename_1 = "{}-{}_{}-strat_{}-explore_{}-runs".format(game_type,game,q_update,exp_strategy_1,run)
    # filename_2 = "{}-{}_{}-strat_{}-explore_{}-runs".format(game_type,game,q_update,exp_strategy_2,run)
    # filename_3 = "{}-{}_{}-strat_{}-explore_{}-runs".format(game_type,game,q_update,exp_strategy_1,run)
    # filename_4 = "{}-{}_{}-strat_{}-explore_{}-runs".format(game_type,game,q_update,exp_strategy_2,run)
    
    # title_1 = filename_1 + tag_1
    # title_2 = filename_2 + tag_2
    # title_3 = filename_3 + tag_3
    # title_4 = filename_4 + tag_4

    ################################################
    
    mean_1, err_up_1, err_down_1 = readGraphData(title_1)
    mean_2, err_up_2, err_down_2 = readGraphData(title_2)
    mean_3, err_up_3, err_down_3 = readGraphData(title_3)
    mean_4, err_up_4, err_down_4 = readGraphData(title_4)
    # mean_5, err_up_5, err_down_5 = readGraphData(title_5)
    
    # mean = [mean_1,mean_2]
    # err_up = [err_up_1,err_up_2]
    # err_down = [err_down_1,err_down_2]
    # label = [label_1,label_2]


    # mean = [mean_1,mean_2,mean_3]
    # err_up = [err_up_1,err_up_2,err_up_3]
    # err_down = [err_down_1,err_down_2,err_down_3]
    # label = [label_1,label_2,label_3]

    
    mean = [mean_1,mean_2,mean_3,mean_4]
    err_up = [err_up_1,err_up_2,err_up_3,err_up_4]
    err_down = [err_down_1,err_down_2,err_down_3,err_down_4]
    label = [label_1,label_2,label_3,label_4]

    
    # mean = [mean_1,mean_2,mean_3,mean_4,mean_5]
    # err_up = [err_up_1,err_up_2,err_up_3,err_up_4,err_up_5]
    # err_down = [err_down_1,err_down_2,err_down_3,err_down_4,err_down_5]
    # label = [label_1,label_2,label_3,label_4,label_5]

    
    # mean = [mean_1]
    # err_up = [err_up_1]
    # err_down = [err_down_1]
    # label = [label_1]
    
    # filename += label_1
    # filename += tag_2

    # filename += "_various_risk_factor"
    
    evalAvg(mean,err_up,err_down,max_r,min_r,
            max_episode,episode_window,filename,save,
            folder,fmt_col,label)

