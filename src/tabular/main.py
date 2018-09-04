from tabular import map_env as me
from tabular import map_stoc_env as ms
from tabular import t_agent as ta
from tabular import helper as hp
import numpy as np

def main():

    game = "risk_windy_maze"          # windy_maze   # hard_windy_maze  # risky_windy_maze
    game_type = "deterministic"                        # deterministic  # stochastic
    start_row = 7
    start_col = 0

    ##### Deterministic Env #########
    
    # maze = me.makeMapEnv(game,start_row,start_col)

    ###### Stochastic Env ############
    
    # anti_stoc_factor = 0.75                                   # the degree of anti-stochasticity
    # maze  = ms.MapStocEnv(game,start_row,start_col,anti_stoc_factor)

    ###### Stochastic Reward Env ######
    
    stoc_state = 2
    stoc_act = 0
    stoc_tres = 0.5
    low_r = 0
    high_r = 4
    maps = None

    goal_state_1 = 6
    goal_act_1 = 2
    # goal_state_2 = 15
    # goal_act_2 = 1

    maze = me.makeMapEnv(game,start_row,start_col,maps,
                         stoc_state,stoc_act,stoc_tres,low_r,high_r)

    ###### Stochastic Reward in Stochastic Env ########
    # maze = ms.MapStocEnv(game,start_row,start_col,anti_stoc_factor,
    #                      maps,stoc_state,stoc_act,stoc_tres,low_r,high_r)

    
    ####################################    
    
    maze.reset()
    maze.render()
    
    ########## Q Learning Params ########
    
    obs_n = maze._obs_space_n
    act_n = maze._agent._action_space_n
    
    discount_factor  = 0.9                          # the discount factor, 0.9 for gauss,epsilon
    learning_rate = 0.1
    learning_decay = 0.5                    # 0.5 for count based # to decay learning rate

    q_update = "risk"                     # vanilla # count # risk
    exp_strategy = "greedy"               # "epsilon", "softmax", "greedy", "boltzmann"
    update_policy = "greedy"               # "epsilon", "greedy", "boltzmann"

    ######### Exploration Strategy #########
    # params below are used interchangeably between epsilon and boltzmann 

    epsilon_type = "constant"           #"linear"   "exponential"   "constant" 
    epsilon_const = 0.1                  # use a constant epsilon policy = 0.5, boltzmann uses 0.1

    epsilon_rate = 0.9            # the polynomial for exponential decay
    max_epsilon = 1.0                      # maximum epsilon value which decays with episodes
    min_epsilon = 0.00001
    diminishing = True               # False to not use the discounted weight for noise in late episodes

    ##########################################
    
    beta_cnt_based = 0.5                      # count-based exploration constant for exploration bonus
    risk_level = 1.0                       # risk seeking level for risk training

    initial_Q = 0.0                       # used 0.0 for risk seeking and epsilon, 0.5 for count
    initial_M = 3.9                       # an example uses 1/(1-discount_factor) for initial_Q
    
    ######### Experiments & Records #########
    """
    pure = pure epsilon greedy exploration linear decay w/o initial optimistic Q

    """
    param_set = "{}_".format(exp_strategy)              # to record different sets of params used
    max_episode = 5000
    run = 1                                 # number of runs to train the agent
    game_step = 100                         # number of game time steps before termination
    no_play = 1                          # number of episodes for the test run
    test_freq = 1                        # frequency of testing, i.e. every nth episode
    monte_freq = 30                       # frequency of monte carlo sampling for each state-action
    monte_test_freq = 1000                  # frequency of checking variance table 

    ########## Saving Files ############
    
    fmt_col = "r"                        # mean line color
    save = False                            # True to save the picture generated from evalEpisode()
    folder = "final"              # windy_maze  # hard_windy_maze

    # tag_1 = "_init_1_{}_epi".format(max_episode)          # label for graph legend
    # tag_1 = "_boltzmann-init-4*"          # label for graph legend
    tag_1 = "_vanilla-risk-7point5*"          # label for graph legend
    filename = "{}-{}_{}-strat_{}-explore_{}-runs".format(game_type,game,q_update,exp_strategy,run)
    label_1 = tag_1[1:]
    mov_title = filename + tag_1      # filename for mov avg data pickle

    # label_1 = exp_strategy
    # mov_title = filename


    ########## Plot Map ##################
    
    vis_file = "{}-{}_{}-strat_{}-explore".format(game_type,game,q_update,exp_strategy)
    plot_type_1 = "val_act"         # "val_act": arrow and val func  # "val_func"  # "act": q-val 
    plot_table_1 = "U"               # U # Q # var # monte_Q # monte_var
    plot_type_2 = "act"         # "val_act": arrow and val func  # "val_func"  # "act": q-val 
    plot_table_2 = "Q"               # U # Q # var # monte_Q # monte_var
    plot_type_3 = "act"         # "val_act": arrow and val func  # "val_func"  # "act": q-val 
    plot_table_3 = "var"               # U # Q # var # monte_Q # monte_var
    
    ####### Moving Average Graph Plotting #######

    episode_window = 500               # size of the window for moving average, use factor of 10
    max_reward = high_r
    max_r = high_r + 0.2                       # upper y bound
    min_r = 0.0                       # lower y bound
    max_var = high_r + 0.1
    min_var = 0.0
    conf_lvl = 0.95                   # confidence level for confidence interval result plotting
    
    ########################################
    
    ################ Q-Learning ##################
    
    # '''

    mov_avg_run = []                          # accumulation of "goals" across multiple runs
    q_est_run = []                       
    q_monte_run = []
    var_est_run = []                      
    var_monte_run = []
    q_delta_run = []
    var_delta_run = []

    
    for run_cnt in range(run):

        t_agent = ta.Tabular_Q_Agent(discount_factor,obs_n,act_n,max_epsilon,min_epsilon,exp_strategy,
                                     diminishing,max_episode,q_update,epsilon_type,epsilon_rate,
                                     epsilon_const,update_policy)
        
        maze.initialiseTable(t_agent.Q,initial_Q)
        maze.initialiseTable(t_agent.M,initial_M)
        t_agent.initialiseU(risk_level)
            
        goals = []                          # accumulation of rewards
        q_est_list = []                        
        q_monte_list = []
        var_est_list = []                      
        var_monte_list = []
        q_delta_list= []
        var_delta_list = []
        done_count  = 0                     # freq of task completion / elimination below max game steps

        for episode in range(max_episode):
            
            state = maze.reset()
            step_count = 0
            
            while step_count <= game_step:

                action = t_agent.act(state,episode)
                new_state, reward, done = maze.step(action,game_step)
                
                learning_rate = t_agent.learningRate(episode,learning_rate,learning_decay)   # can define power value
                                                                               # for diff decay rate
                if(q_update == "vanilla"):
                    t_agent.train(new_state,reward,state,action,learning_rate)           # normal training
                elif(q_update == "count"):
                    t_agent.count_train(new_state,reward,state,action,
                                        learning_rate,beta_cnt_based)   # count-based training
                elif(q_update == "risk"):                               # risk training
                    t_agent.risk_train(new_state,reward,state,action,risk_level,episode,learning_rate)
                
                # if episode > 100 and episode < 110:                   
                #     print("Action = ",action)
                #     maze.render()
                #     print("Q {} epi = \n".format(episode))
                #     print(np.array_str(t_agent.Q,precision=2,suppress_small=True))
                    
                #     print("Var {} epi = \n".format(episode))
                #     print(np.array_str(t_agent.var,precision=2,suppress_small=True))
                
                #     print("U {} epi  = \n".format(episode))
                #     print(np.array_str(t_agent.U,precision=2,suppress_small=True))


                state = new_state 
                step_count+=1
            
                if done == True:
                    done_count += 1
                    break

            actual_goals = hp.playGame(t_agent,maze,game_step,no_play,episode,max_episode)
            actual_avg = sum(actual_goals)/no_play
            goals.append(actual_avg)

            if episode == 0:
                hp.plotMap(t_agent,maze,plot_table_1,"val_func",vis_file,episode)
                
            # if episode % monte_test_freq == 0:

            #     hp.monteCarlo(t_agent,maze,game_step,monte_freq,discount_factor)

            #     q_monte, q_est = hp.meanMonte(t_agent.monte_goal,t_agent.Q)
            #     var_monte, var_est = hp.meanMonte(t_agent.monte_var,t_agent.var)
            #     q_delta_full, q_delta = hp.monteDiff(t_agent.monte_goal,t_agent.Q)
            #     var_delta_full, var_delta = hp.monteDiff(t_agent.monte_var,t_agent.var)
                
            #     q_est_list.append(q_est)
            #     q_monte_list.append(q_monte)
            #     var_est_list.append(var_est)
            #     var_monte_list.append(var_monte)
            #     q_delta_list.append(q_delta)
            #     var_delta_list.append(var_delta)
                
                ############ Visual Map ##############

                # if episode > 200:
                # hp.plotMap(t_agent,maze,plot_table_1,plot_type_1,vis_file,episode)
                # hp.plotMap(t_agent,maze,plot_table_2,plot_type_2,vis_file,episode)
                # hp.plotMap(t_agent,maze,plot_table_3,plot_type_3,vis_file,episode)
                    
                ########### Test Print #################
                
                # hp.plotMap(t_agent,maze,t_agent.Q,"val_func")
                                
                # last_epi = max_episode - episode
                # if(last_epi <= 10):
                #     hp.plotMap(t_agent,maze,plot_table_3,plot_type_3,vis_file,episode)
            
                # print("Last {} Q Table  = \n".format(last_epi))
                # print(np.array_str(t_agent.Q,precision=2,suppress_small=True))
                # print("Last {} Variance = \n".format(last_epi))
                # print(np.array_str(t_agent.var,precision=2,suppress_small=True))
                

                # print("Final Monte Q = \n")
                # print(np.array_str(t_agent.monte_goal,precision=2,suppress_small=True))

                # print("Action = ",action)
                # maze.render()
                
                # print("Final Q Table  = \n")
                # print(np.array_str(t_agent.Q,precision=2,suppress_small=True))

                # print("Final Var \n")
                # print(np.array_str(t_agent.var,precision=2,suppress_small=True))
                

                # print("Final U Table  = \n")
                # print(np.array_str(t_agent.U,precision=2,suppress_small=True))


                
                # print("Final Monte Var = \n")
                # print(np.array_str(t_agent.monte_var,precision=2,suppress_small=True))

                # print("Final Count Table  = ")
                # print(np.array_str(t_agent.visit_count,suppress_small=True))
                
                    
                # print("Final M Table  = \n")
                # print(np.array_str(t_agent.M,precision=2,suppress_small=True))
            

                
        ########## result for Each Training Run #############

        hp.plotMap(t_agent,maze,plot_table_1,plot_type_1,vis_file,episode)
        hp.plotMap(t_agent,maze,plot_table_2,plot_type_2,vis_file,episode)
        hp.plotMap(t_agent,maze,plot_table_3,plot_type_3,vis_file,episode)
        
        
        ########## Monte Carlo Comparison ####################

        # q_est_run.append(q_est_list)                        
        # q_monte_run.append(q_monte_list)                        
        # var_est_run.append(var_est_list)                                              
        # var_monte_run.append(var_monte_list)                        
        # q_delta_run.append(q_delta_list)                        
        # var_delta_run.append(var_delta_list)                        
        
        q_label = "Q"
        var_label = "Var"
        q_monte_title = q_label + "_delta_" + filename
        var_monte_title = var_label + "_delta_" + filename
        # hp.evalMonte(q_est_list,q_monte_list,max_episode,monte_test_freq,q_monte_title,q_label,q_delta_list)
        # hp.evalMonte(var_est_list,var_monte_list,max_episode,monte_test_freq,var_monte_title,var_label,var_delta_list)
       
        # hp.monteCarlo(t_agent,maze,game_step,monte_freq,discount_factor)
        
        ########## Test Print ##########
        
        # print("Final Monte Q = \n")
        # print(np.array_str(t_agent.monte_goal,precision=2,suppress_small=True))
        
        # print("Final Q Table  = \n")
        # print(np.array_str(t_agent.Q,precision=2,suppress_small=True))
        
        # print("Final Monte Var = \n")
        # print(np.array_str(t_agent.monte_var,precision=2,suppress_small=True))

        # print("Final Var \n")
        # print(np.array_str(t_agent.var,precision=2,suppress_small=True))
 
        # print("Final U Table  = \n")
        # print(np.array_str(t_agent.U,precision=2,suppress_small=True))

        # print("Final M Table  = \n")
        # print(np.array_str(t_agent.M,precision=2,suppress_small=True))

        # print("Final Count Table  = ")
        # print(np.array_str(t_agent.visit_count,suppress_small=True))
        # print("No. of plays under {0} game steps = ".format(game_step),done_count)

        print("Count to R =")
        print(t_agent.visit_count[stoc_state,stoc_act],"\n")
        print ("Count to G =")
        print(t_agent.visit_count[goal_state_1,goal_act_1],"\n")
        
        #####################################################################
        
        no_testing = max_episode/test_freq
        # print("Average goal collected for each episode of test play:",goals)
        avg_score = sum(goals)/no_testing
        
        print("Average score across testing episodes:", avg_score)
        maze.render()
        print("Current run count = ",run_cnt)

        
        ############### Calc the Moving Average of Rewards ####################

        # mov_avg = hp.calcMovingAverage(goals,episode_window)
        # mov_avg_run.append(mov_avg)
        
        # hp.evalEpisode(mov_avg,max_episode,episode_window,filename)     # to print the current run mov avg
        

    ######################### end of Multiple Runs ########################################

    # mov_data = hp.confInterval(mov_avg_run,conf_lvl,max_reward)
 
    # mean_data = [mov_data[0]]
    # err_up_data = [mov_data[1]]
    # err_down_data = [mov_data[2]]
    # label_data= [label_1]
    
    # hp.evalAvg(mean_data,err_up_data,err_down_data,max_r,min_r,
    #            max_episode,episode_window,filename,save,
    #            folder,fmt_col,label_data)

    # hp.saveGraphData(mov_data,mov_title) 


    ################## delta Eval ##############################
    
    # mov_q_est = hp.confInterval(q_est_run,conf_lvl,max_reward)
    # mov_q_monte = hp.confInterval(q_monte_run,conf_lvl,max_reward)
    # mov_q_delta = hp.confInterval(q_delta_run,conf_lvl,max_reward)
    # mov_var_est = hp.confInterval(var_est_run,conf_lvl,initial_M)
    # mov_var_monte = hp.confInterval(var_monte_run,conf_lvl,initial_M)
    # mov_var_delta = hp.confInterval(var_delta_run,conf_lvl,initial_M)

    # hp.plotDeltaRun(mov_q_est,mov_q_monte,mov_q_delta,max_r,min_r,
    #                 max_episode,monte_test_freq,q_monte_title,save,
    #                 folder,fmt_col,q_label)

    # hp.plotDeltaRun(mov_var_est,mov_var_monte,mov_var_delta,max_var,min_var,
    #                 max_episode,monte_test_freq,var_monte_title,save,
    #                 folder,fmt_col,var_label)


    # '''  

    '''
    ############## Store the Result ###############
    
    if q_update == "count":
        params = [max_episode,discount_factor,learning_rate,beta_cnt_based,initial_Q,
                  run_cnt,avg_score]
        string_param = "max_episode,discount_factor,learning_rate,beta_cnt_based,initial_Q,run_cnt,avg_score"
        table = [t_agent.Q]
        table_param = "Q-Table"
    
    
    elif q_update == "risk":
        params = [max_episode,discount_factor,learning_rate,risk_level,initial_Q,
                  initial_M,initial_U,run_cnt,avg_score]
        string_param = "max_episode,discount_factor,learning_rate,risk_level,initial_Q,initial_M,initial_U,run_cnt,avg_score"
        table = [t_agent.Q,t_agent.U]
        table_param = "Q-Table, U-Table"
    
    
    elif q_update == "vanilla":
        params = [max_episode,discount_factor,learning_rate,initial_Q,run_cnt,avg_score]
        string_param = "max_episode,discount_factor,learning_rate,initial_Q,run_cnt,avg_score"
        table = [t_agent.Q]
        table_param = "Q-Table"
    
    hp.writeResult(filename,folder,params,string_param,run_cnt)
    hp.storeTable(filename,folder,table,table_param,run_cnt)
    

    ######### Random Action Sampling ##########
    
    t_agent = ta.Tabular_Q_Agent(discount_factor,obs_n,act_n,max_epsilon,min_epsilon,exp_strategy,
                                 diminishing,max_episode,q_update,epsilon_type,epsilon_rate,
                                 epsilon_const,update_policy)
    num_epi = max_episode
    game_cnt = game_step
    run_count = run
    mov_avg_run = []
    
    for i in range(run_count):
        rand_rewards = hp.playGame(t_agent,maze,game_cnt,num_epi,"rand")    # self-created env # deprecated

        rand_avg_rewards = sum(rand_rewards)/num_epi
        print(rand_avg_rewards)
        
        mov_avg = hp.calcMovingAverage(rand_rewards,episode_window)
        mov_avg_run.append(mov_avg)
        
        # hp.evalEpisode(mov_avg,max_episode,episode_window,filename)     # to print the current run mov avg
    
  
    '''

if __name__ == "__main__":
    main()
    
