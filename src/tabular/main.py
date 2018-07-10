from tabular import map_env as me
from tabular import t_agent as ta
from tabular import helper as hp
        

def main():

    game = "windy_maze"          # windy_maze   # hard_windy_maze
    maze = me.makeMapEnv(game) 

    ####### Q Parameters ##########
    
    obs_n = maze._obs_space_n
    act_n = maze._agent._action_space_n
    
    discount_factor  = 0.9                          # the discount factor, 0.9 for gauss,epsilon
    learning_decay = 0.5                            # 0.5 for count based # to decay learning rate
    
    q_update = "epsilon"                    # epsilon # count # risk # q-update method
    explore_method = "epsilon"           # "epsilon", "risk", True (gauss), False (greedy)
    epsilon_decay_type = "linear"           # "linear"   "exponential"
    epsilon_decay_rate = 0.3
    
    max_epsilon = 1.0                      # maximum epsilon value which decays with episodes
    min_epsilon = 0.00001
    diminishing_weight = True            # False to not use the discounted weight for noise in late episodes

    beta_cnt_based = 0.5                      # count-based exploration constant for exploration bonus
    risk_level = 0.2                       # risk seeking level for risk training

    initial_Q = 0.0                       # used 0.0 for risk seeking and epsilon, 0.5 for count
    initial_M = 1.0                       # an example uses 1/(1-discount_factor) for initial_Q
    
    ######### Experiments & Records #########
    """
    pure = pure epsilon greedy exploration linear decay w/o initial optimistic Q

    """
    param_set = "epsilon_risk_0.2"              # to record different sets of params used
    max_episode = 1000
    run = 1                                 # number of runs to train the agent
    game_step = 100                         # number of game time steps before termination
    no_play = 30                          # number of episodes for the test run
    test_freq = 1                        # frequency of testing, i.e. every nth episode
    
    save = False                            # True to save the picture generated from evalEpisode()
    folder = "hard_windy_maze"              # windy_maze  # hard_windy_maze
    game_type = "dtm"                        # dtm  # stoc
    filename = "{}_{}_exp_{}_runs_{}_epi_type_{}".format(game_type,q_update,run,max_episode,param_set)

    ####### Moving Average Graph Plotting #######

    episode_window = 100                     # size of the window for moving average, use factor of 10
    max_reward = 1.0
    max_r = 1.2                           # upper y bound
    min_r = 0.0                           # lower y bound
    conf_lvl = 0.95                         # confidence level for confidence interval result plotting
    
    ########################################

    ######### Random Action Sampling ##########
    '''
    t_agent = ta.Tabular_Q_Agent(discount_factor,obs_n,act_n,max_epsilon,min_epsilon,explore_method,
                                 diminishing_weight,max_episode,q_update,epsilon_decay_rate)

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

    mov_avg_run = []                          # accumulation of "goals" across multiple runs
    
    for run_cnt in range(run):

        t_agent = ta.Tabular_Q_Agent(discount_factor,obs_n,act_n,max_epsilon,min_epsilon,explore_method,
                                     diminishing_weight,max_episode,q_update,epsilon_decay_rate)
        t_agent.Q[t_agent.Q == 0] = initial_Q
        t_agent.M[t_agent.M == 0] = initial_M

        goals = []                          # accumulation of rewards
        done_count  = 0                     # freq of task completion / elimination below max game steps

        for episode in range(max_episode):
            
            state = maze.reset()
            step_count = 0
            
            while step_count <= game_step:

                action = t_agent.act(state,episode,epsilon_decay_type)
                new_state, reward, done = maze.step(action)
                
                learning_rate = t_agent.learningRate(episode,learning_decay)   # can define power value
                                                                               # for diff decay rate
                if(q_update == "epsilon"):
                    t_agent.train(new_state,reward,state,action,learning_rate)           # normal training
                elif(q_update == "count"):
                    t_agent.count_train(new_state,reward,state,action,
                                        learning_rate,beta_cnt_based)   # count-based training
                elif(q_update == "risk"):
                    t_agent.risk_train(new_state,reward,state,action,risk_level,episode,epsilon_decay_type,
                                       learning_rate)  # risk-seeking training # make sure it's on-policy,
                                                       # i.e. change the risk_train code
                state = new_state
                step_count+=1
            
                if done == True:
                    done_count += 1
                    break
                
        ############ Using Current Q Table to Play Games without further Update ##################
        
            if(episode % test_freq == 0):
                actual_goals = hp.playGame(t_agent,maze,game_step,no_play)
                actual_avg = sum(actual_goals) /no_play
                goals.append(actual_avg)

        ########## Result for Each Training Run #############

        # print("Final M Table  = \n",t_agent.M)
        print("Final Q Table  = \n",t_agent.Q)
        # print("Final U Table  = \n",t_agent.U)
        # print("Final Count Table  = \n",t_agent.visit_count)
        print("No. of plays under {0} game steps = ".format(game_step),done_count)
        
        no_testing = max_episode/test_freq
        assert(no_testing == len(goals))
        print("Average goal collected for each episode of test play:",goals)
        avg_score = sum(goals)/no_testing
        
        print("Average score every {} episode:".format(test_freq), avg_score)
        # maze.render()
        print("Current run count = ",run_cnt)
        
        # '''

        ############## Store the Result ###############
                
        # if q_update == "count":
        #     params = [max_episode,discount_factor,learning_decay,beta_cnt_based,initial_Q,
        #               run_cnt,avg_score]
        #     string_param = "max_episode,discount_factor,learning_decay,beta_cnt_based,initial_Q,run_cnt,avg_score"
        #     table = [t_agent.Q]
        #     table_param = "Q-Table"

            
        # elif q_update == "risk":
        #     params = [max_episode,discount_factor,learning_decay,risk_level,initial_Q,
        #               initial_M,initial_U,run_cnt,avg_score]
        #     string_param = "max_episode,discount_factor,learning_decay,risk_level,initial_Q,initial_M,initial_U,run_cnt,avg_score"
        #     table = [t_agent.Q,t_agent.U]
        #     table_param = "Q-Table, U-Table"

            
        # elif q_update == "epsilon":
        #     params = [max_episode,discount_factor,learning_decay,max_epsilon,min_epsilon,
        #               initial_Q,run_cnt,avg_score]
        #     string_param = "max_episode,discount_factor,learning_decay,max_epsilon,min_epsilon,initial_Q,run_cnt,avg_score"
        #     table = [t_agent.Q]
        #     table_param = "Q-Table"

        # hp.writeResult(filename,folder,params,string_param,run_cnt)
        # hp.storeTable(filename,folder,table,table_param,run_cnt)
        
        ############### Calc the Moving Average of Rewards ####################

        mov_avg = hp.calcMovingAverage(goals,episode_window)
        mov_avg_run.append(mov_avg)
        
        # hp.evalEpisode(mov_avg,max_episode,episode_window,filename)     # to print the current run mov avg
        

    ######################### End of Multiple Runs ########################################
    
    mean_mov_avg, err_mov_avg = hp.confInterval(mov_avg_run,conf_lvl,max_reward)
    
    hp.avgEvalEpisode(mean_mov_avg,err_mov_avg,max_r,min_r,
                   max_episode,episode_window,filename,save,folder)

                                        
                                        
if __name__ == "__main__":
    main()
