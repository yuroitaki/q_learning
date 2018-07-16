import numpy as np
import gym as me
from gym.envs.registration import register

from tabular import t_agent as ta
from tabular import helper_gym as hp

# np.set_printoptions(threshold=np.nan)

# '''
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

register(
    id='BigFrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '8x8', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

# '''

def main():

    game = "Taxi-v2"         # FrozenLake-v0  # FrozenLakeNotSlippery-v0  # CliffWalking-v0
    maze = me.make(game) 

    ####### Q Parameters ##########
    
    obs_n = maze.observation_space.n               # openAI gym env   
    act_n = maze.action_space.n
    
    discount_factor  = 0.9                          # the discount factor, 0.9 for gauss,epsilon
    learning_decay = 0.5                    # 0.5 for count based # to decay learning rate

    q_update = "risk"                     # epsilon # count # risk
    exp_strategy = "epsilon"               # "epsilon", "softmax", "greedy"
    update_policy = "greedy"               # "epsilon", "softmax", "greedy"
    
    epsilon_type = "constant"           # "linear"   "exponential"   "constant"
    epsilon_const = 0.5                 # use a constant epsilon policy

    epsilon_rate = 0.3               # the polynomial for exponential decay
    max_epsilon = 1.0                      # maximum epsilon value which decays with episodes
    min_epsilon = 0.00001
    diminishing = True            # False to not use the discounted weight for noise in late episodes

    beta_cnt_based = 0.5                      # count-based exploration constant for exploration bonus
    risk_level = 0.3                       # risk seeking level for risk training

    initial_Q = 0.0                       # used 0.0 for risk seeking and epsilon, 0.5 for count
    initial_M = 1.0                       # an example uses 1/(1-discount_factor) for initial_Q
    initial_U = 1.0
    
    ######### Experiments & Records #########
    """
    pure = pure epsilon greedy exploration linear decay w/o initial optimistic Q

    """
    param_set = "{}_".format(exp_strategy)              # to record different sets of params used
    max_episode = 500
    run = 20                                 # number of runs to train the agent
    game_step = 200                         # number of game time steps before termination
    no_play = 1                          # number of episodes for the test run
    test_freq = 1                        # frequency of testing, i.e. every nth episode
    
    save = False                            # True to save the picture generated from evalEpisode()
    folder = "frozen_lake_not_slippery"              # frozen_lake  # frozen_lake_not_slippery  # cliff_walking
    game_type = "dtm"                        # dtm  # stoc
    filename = "{}_{}_exp_{}_runs_{}_epi_type_{}".format(game_type,q_update,run,max_episode,param_set)

    ####### Moving Average Graph Plotting #######

    episode_window = 100                     # size of the window for moving average, use factor of 10
    max_reward = 20.0                     # max reward for the game
    max_r = 1.2                           # upper y bound
    min_r = 0.0                           # lower y bound
    conf_lvl = 0.95                         # confidence level for confidence interval result plotting
    
    ########################################

    ######### Random Action Sampling ##########
    '''
    t_agent = ta.Tabular_Q_Agent(discount_factor,obs_n,act_n,max_epsilon,min_epsilon,exp_strategy,
                                 diminishing,max_episode,q_update,epsilon_type,epsilon_rate,
                                 epsilon_const,update_policy)
    num_epi = 1000
    game_cnt = 100
    run_count = 1
    
    for i in range(run_count):
        rand_rewards = hp.playGame(t_agent,maze,game_cnt,"rand")    
        rand_avg_rewards = sum(rand_rewards)/num_epi
        print(rand_avg_rewards)
    '''

    ################ Q-Learning ##################
    
    # '''

    mov_avg_run = []                          # accumulation of "goals" across multiple runs
    
    for run_cnt in range(run):

        t_agent = ta.Tabular_Q_Agent(discount_factor,obs_n,act_n,max_epsilon,min_epsilon,exp_strategy,
                                     diminishing,max_episode,q_update,epsilon_type,epsilon_rate,
                                     epsilon_const,update_policy)

        t_agent.Q[t_agent.Q == 0] = initial_Q
        t_agent.M[t_agent.M == 0] = initial_M
        t_agent.U[t_agent.U == 0] = initial_U

        goals = []                          # accumulation of rewards
        done_count  = 0                     # freq of task completion / elimination below max game steps

        for episode in range(max_episode):
            
            state = maze.reset()
            step_count = 0
            acc_reward = 0

            while step_count <= game_step:

                action = t_agent.act(state,episode)

                new_state, reward, done, info = maze.step(action)
                learning_rate = t_agent.learningRate(episode,learning_decay)   # can define power value
                                                                               # for diff decay rate
                if(q_update == "epsilon"):
                    t_agent.train(new_state,reward,state,action,learning_rate)           # normal training
                elif(q_update == "count"):
                    t_agent.count_train(new_state,reward,state,action,
                                        learning_rate,beta_cnt_based)   # count-based training
                elif(q_update == "risk"):                               # risk training
                    t_agent.risk_train(new_state,reward,state,action,risk_level,episode,learning_rate)
                                                                         
                state = new_state
                step_count += 1
                acc_reward += reward
                
                if done == True:
                    done_count += 1
                    break
                
        ############ Using Current Q Table to Play Games without further Update ##################
        
            # if(episode % test_freq == 0):
            actual_goals = hp.playGame(t_agent,maze,game_step,no_play,episode,max_episode)
            actual_avg = sum(actual_goals) /no_play
            goals.append(actual_avg)

        ########## Result for Each Training Run #############

        # print("Final Q Table  = \n",t_agent.Q)
        # print("Final M Table  = \n",t_agent.M)
        print("Final U Table  = \n",t_agent.U)
        # print("Final Count Table  = ")
        # print(np.array_str(t_agent.visit_count,suppress_small=True))
        print("No. of plays under {0} game steps = ".format(game_step),done_count)
        
        no_testing = max_episode/test_freq
        # print("Average goal collected for each episode of test play:",goals)
        avg_score = sum(goals)/no_testing
        
        print("Average score across testing episodes:", avg_score)
        maze.render()
        print("Current run count = ",run_cnt)

        # '''

        ############## Store the Result ###############
                
        # if q_update == "count":
        #     params = [max_episode,discount_factor,learning_rate,beta_cnt_based,initial_Q,
        #               run_cnt,avg_score]
        #     string_param = "max_episode,discount_factor,learning_rate,beta_cnt_based,initial_Q,run_cnt,avg_score"
        #     table = [t_agent.Q]
        #     table_param = "Q-Table"

            
        # elif q_update == "risk":
        #     params = [max_episode,discount_factor,learning_rate,risk_level,initial_Q,
        #               initial_M,initial_U,run_cnt,avg_score]
        #     string_param = "max_episode,discount_factor,learning_rate,risk_level,initial_Q,initial_M,initial_U,run_cnt,avg_score"
        #     table = [t_agent.Q,t_agent.U]
        #     table_param = "Q-Table, U-Table"

            
        # elif q_update == "epsilon":
        #     params = [max_episode,discount_factor,learning_rate,initial_Q,run_cnt,avg_score]
        #     string_param = "max_episode,discount_factor,learning_rate,initial_Q,run_cnt,avg_score"
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
