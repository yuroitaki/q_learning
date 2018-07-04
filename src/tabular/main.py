from tabular import map_env as me
from tabular import t_agent as ta
from tabular import helper as hp
        

def main():

    game = "windy_maze"          # windy_maze   # hard_windy_maze
    maze = me.makeMapEnv(game) 

    ####### Q Parameters ##########
    
    obs_n = maze._obs_space_n
    act_n = maze._agent._action_space_n
    
    gamma  = 0.9                          # 0.9 for gauss,epsilon
    learning_rate = 0.8                   # 0.8 for gauss, epsilon
    
    exploration = "epsilon"                    # epsilon # count # risk
    discount_noise = True           # "epsilon_lin/epsilon_exp", "risk", True (gauss), False (greedy)    
    max_epsilon = 1.0                      # maximum epsilon value which decays with episodes
    min_epsilon = 0.001
    diminishing_weight = True            # False to not use the discounted weight for noise in late episodes

    beta_cnt_based = 0.3                      # count-based exploration constant for exploration bonus
    risk_level = 0.2                       # risk seeking level for risk training
    initial_Q = 0.0                       # used 0.0 for risk seeking and epsilon, 0.4 for count
    initial_M = 1.0                       # an example uses 1/(1-gamma) for initial_Q
    
    ######### Experiments & Records #########

    param_set = ""                          # alphanumeric number to record different sets of params used
    max_episode = 100
    run = 10                                 # Number of runs to train the agent
    game_step = 100
    
    save = False                            # True to save the picture generated from evalEpisode()
    folder = "hard_windy_maze"              # windy_maze  # hard_windy_maze
    game_type = "deterministic"              # deterministic  # stochastic
    filename = "{}_{}_exploration_{}_episodes_type_{}".format(game_type,exploration,max_episode,param_set)

    ####### Moving Average Graph Plotting #######

    episode_window = 1                     # size of the window for moving average
    max_reward = 1.0
    max_r = 1.2                           # upper y bound
    min_r = 0.0                           # lower y bound
    conf_lvl = 0.95                         # confidence level for confidence interval result plotting
    
    ########################################

    t_agent = ta.Tabular_Q_Agent(gamma,learning_rate,obs_n,act_n,
                                 max_epsilon,min_epsilon,discount_noise,
                                 diminishing_weight,max_episode)

    ######### Random Action Sampling ##########
    '''
    num_epi = 1000
    game_cnt = 100
    run_count = 1
    
    for i in range(run_count):
        rand_rewards = hp.playGame(t_agent,maze,game_cnt,"rand")    # self-created env
        rand_avg_rewards = sum(rand_rewards)/num_epi
        print(rand_avg_rewards)
    '''

    ################ Q-Learning ##################
    
    # '''
    t_agent.Q[t_agent.Q == 0] = initial_Q
    t_agent.M[t_agent.M == 0] = initial_M

    mov_avg_run = []                          # accumulation of "goals" across multiple runs
    
    for run_cnt in range(run):

        goals = []                          # accumulation of rewards
        done_count  = 0                     # freq of task completion / elimination below max game steps

        for episode in range(max_episode):
            
            state = maze.reset()
            acc_reward = 0
            step_count = 0
            
            while step_count <= game_step:

                action = t_agent.act(state,episode)
                new_state, reward, done = maze.step(action)

                if(exploration == "epsilon"):
                    t_agent.train(new_state,reward,state,action)                     # normal training
                elif(exploration == "count"):
                    t_agent.count_train(new_state,reward,state,action,beta_cnt_based)   # count-based training
                elif(exploration == "risk"):
                    t_agent.risk_train(new_state,reward,state,action,risk_level)      # risk-seeking training
                
                acc_reward += reward
                state = new_state
                step_count+=1
            
                if done == True:
                    done_count += 1
                    break
                
            goals.append(acc_reward)

        ########## Result for Each Training Run #############
            
        # print("Final Q Table  = \n",t_agent.Q)
        # print("Final U Table  = \n",t_agent.U)
        # print("Final Count Table  = \n",t_agent.visit_count)
        # print("No. of plays under {0} game steps = ".format(game_step),done_count)
        avg_score = sum(goals)/max_episode
        # print("Average score per episode:", avg_score)
        maze.render()

        ############ Using Final Q Table to Play Games without further Update ##################

        actual_goals = hp.playGame(t_agent,maze,game_step)
        actual_avg = sum(actual_goals)/max_episode
        # print("Average actual score per episode:", actual_avg)
        
        # '''

        ############## Store the Result ###############
                
        # if exploration == "count":
        #     params = [max_episode,game_step,gamma,learning_rate,beta_cnt_based,initial_Q,
        #               run_cnt,avg_score,actual_avg]
        #     string_param = "max_episode,game_step,gamma,learning_rate,beta_cnt_based,initial_Q,run_cnt,avg_score,actual_avg"
        #     table = [t_agent.Q]
        #     table_param = "Q-Table"

            
        # elif exploration == "risk":
        #     params = [max_episode,game_step,gamma,learning_rate,risk_level,initial_Q,
        #               initial_M,run_cnt,avg_score,actual_avg]
        #     string_param = "max_episode,game_step,gamma,learning_rate,risk_level,initial_Q,initial_M,run_cnt,avg_score,actual_avg"
        #     table = [t_agent.Q,t_agent.U]
        #     table_param = "Q-Table, U-Table"

            
        # elif exploration == "epsilon":
        #     params = [max_episode,game_step,gamma,learning_rate,max_epsilon,min_epsilon,
        #               initial_Q,run_cnt,avg_score,actual_avg]
        #     string_param = "max_episode,game_step,gamma,learning_rate,max_epsilon,min_epsilon,initial_Q,run_cnt,avg_score,actual_avg"
        #     table = [t_agent.Q]
        #     table_param = "Q-Table"


        # hp.writeResult(filename,folder,params,string_param,run_cnt)
        # hp.storeTable(filename,folder,table,table_param,run_cnt)
        
        # ############### Calc the Moving Average of Rewards ####################

        mov_avg = hp.calcMovingAverage(goals,episode_window)
        mov_avg_run.append(mov_avg)
        
        # hp.evalEpisode(mov_avg,max_episode,episode_window,filename)     # to print the current run mov avg
        
        ############################################################################

    ########################### End of Multiple Runs ########################################

    mean_mov_avg, err_mov_avg = hp.confInterval(mov_avg_run,conf_lvl,max_reward)
    
    hp.avgEvalEpisode(mean_mov_avg,err_mov_avg,max_r,min_r,
                   max_episode,episode_window,filename,save,folder)

                                        
                                        
if __name__ == "__main__":
    main()
