import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.envs.registration import register
from tabular import t_agent as ta

# '''
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)
# '''

def gymRandom(maze,max_epi,game_step):
    
    goals = []
    
    for episode in range(max_epi):        
        maze.reset()
        step_count  = 0
        
        while step_count <= game_step:
            
            action = maze.action_space.sample()
            new_state, reward, done, info = maze.step(action)
            step_count += 1
            
            if(done == True):
                break
        goals.append(reward)
        
    return goals



def playGame(t_agent,maze,game_step):
    
    goals = []
    
    for episode in range(t_agent.max_epi):        
        state = maze.reset()
        acc_reward = 0
        step_count = 0

        while step_count <= game_step:

            action = t_agent.play(state)
            new_state, reward, done, info = maze.step(action)
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
        fig.savefig("/vol/bitbucket/ttc14/thesis/fig/{0}/{1}.png".format(folder,title),dpi=100)
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

    
def writeResult(filename,folder,params,run):
    
    with open("/vol/bitbucket/ttc14/thesis/result/{0}/{1}.txt".format(folder,filename),"a+") as f:
        if(run==0):
            f.write("no_of_episodes gamma learning_rate discount_noise punish ith_run avg_score actual_avg\n")
        for item in params:
            f.write(str(item))
            f.write(" ")
        f.write("\n")
        

def main():

    game = "FrozenLakeNotSlippery-v0"   # FrozenLake-v0  # FrozenLakeNotSlippery-v0  # CliffWalking-v0
    maze = gym.make(game)               # openAI gym env
    
    ###### Random Action Sampling ########
    '''
    num_epi = 100
    game_cnt = 100
    run_cnt = 1
    
    for i in range(run_cnt):
        rewards = gymRandom(maze,num_epi,game_cnt)           # openAI gym env
        avg_rewards = sum(rewards)/num_epi
        print(avg_rewards)
    '''
    
    ############ Q-Learning ###############

    ####### Q Parameters ##########
    
    obs_n = maze.observation_space.n               # openAI gym env   
    act_n = maze.action_space.n
    gamma  = 0.9
    learning_rate = 0.8
    max_epsilon = 1.0                      # maximum epsilon value which decays linearly with episodes
    min_epsilon = 0.001
    discount_noise = False                # False to use epsilon-greedy
    diminishing_weight = True              # False to not use the discounted weight for noise in late episodes
    
    ####### Experiment Freq ######
    
    max_episode = 1000
    game_step = 100
    run = 1                                 # Number of runs to train the agent 
    save = False                            # True to save the picture generated from evalEpisode()
    episode_window = 100                     # size of the window for moving average
    folder = "frozen_lake"           # frozen_lake  # frozen_lake_not_slippery  # cliff_walking

    #################################
    # '''
    t_agent = ta.Tabular_Q_Agent(gamma,learning_rate,obs_n,
                                 act_n,max_epsilon,min_epsilon,discount_noise,
                                 diminishing_weight,max_episode)

    for i in range(run):

        goals = []                          # accumulation of rewards
        done_count  = 0                     # freq of task completion / elimination below max game steps
        action_count = []                   # freq of greedy actions

        for episode in range(max_episode):
            
            state = maze.reset()
            acc_reward = 0
            step_count = 0
            act_count = 0
            
            while step_count <= game_step:

                action = t_agent.act(state,episode)
                if(action == np.argmax(t_agent.Q[state,:])):
                    act_count+=1
                    
                new_state, reward, done, info = maze.step(action)
                t_agent.train(new_state,reward,state,action)
                
                acc_reward += reward
                state = new_state
                step_count+=1
            
                if done == True:
                    done_count += 1
                    break
                
            goals.append(acc_reward)
            action_count.append(act_count)

        ########## Result for Each Training Run #############
            
        print("Final Q Table  =",t_agent.Q)
        print("No. of plays under {0} game steps = ".format(game_step),done_count)
        # print("Greedy action count =",sum(action_count)/len(action_count),action_count)
        print("Rewards collected:", goals)
        avg_score = sum(goals)/max_episode
        print("Average score per episode:", avg_score)
        maze.render()

        ############ Using Final Q Table to Play Games without further Update ##################

        actual_goals = playGame(t_agent,maze,game_step)
        actual_avg = sum(actual_goals)/max_episode
        print("Average actual score per episode:", actual_avg)
        
        # '''

        ############## Store the Result ###############
        # '''
        # filename = "Tabular_QLearning_Result_of_{0}_{1}_episodes".format(game,max_episode) 
        # params = [max_episode,gamma,learning_rate,discount_noise,i,avg_score,actual_avg]
        
        # writeResult(filename,folder,params,i)
        # '''
        
        # ############### Plot the Change of Goal against Episode ####################
        '''
        title = "{0}th_Tabular_QLearning_Result_of_{1}_{2}_episodes".format(i,game,max_episode)
        # goals = [1,2,3,4,5,6]
        evalEpisode(goals,max_episode,episode_window,title,save,folder)
        '''


 
if __name__ == "__main__":
    main()