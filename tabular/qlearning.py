import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)


def randomSampling(env,num_episode,game_step):
    
    goals = []
    for episode in range(num_episode):
        env.reset()
        count  = 0
        while count <= game_step:
            action = env.action_space.sample()
            new_state, reward, done, info = env.step(action)
            count += 1
            env.render()
            if(done == True):
                break
        goals.append(reward)
    return goals
    

def qLearning(env,num_episode,gamma,learning_rate,game_step,punish,discount_noise,punish_val,diminishing_weight):
    
    Q = np.zeros([env.observation_space.n,env.action_space.n])
    goals = []
    counter  = 0
    for episode in range(num_episode):
        state = env.reset()
        acc_reward = 0
        count = 0
        
        while count <= game_step:
            if discount_noise == True and diminishing_weight == True:
                action = np.argmax(Q[state,:]+np.random.randn(1,env.action_space.n)*(1/(episode+1)))
            elif discount_noise == True and diminishing_weight == False:
                action = np.argmax(Q[state,:]+np.random.randn(1,env.action_space.n))
            else:
                action = np.argmax(Q[state,:])
                
            new_state, reward, done, info = env.step(action)
            if (punish == True) and (done == True) and (reward == 0):
                reward = punish_val

            optimal_Q = np.max(Q[new_state,:])
            Q[state,action] = Q[state,action] + learning_rate*(reward + gamma*(optimal_Q) - Q[state,action])
            acc_reward += reward
            state = new_state
            count+=1
            
            if done == True:
                counter += 1
                break
        if(acc_reward==punish_val):
            acc_reward = 0
        goals.append(acc_reward)
        
    print("Final Q Table  =",Q)
    print("No. of plays under 100 game steps = ",counter)
    env.render()
    return goals, Q


def evalEpisode(score,num_episode,title,save,folder):
    
    fig = plt.figure(figsize=(32,16))
    x = [i+1 for i in range(num_episode)]
    plt.scatter(x,score,marker='x')
    plt.title(title,fontweight='bold')
    plt.xlabel("Episode No.")
    plt.ylabel("Average Score")
    plt.show()
    if save == True:
        fig.savefig("/vol/bitbucket/ttc14/thesis/fig/{0}/{1}.png".format(folder,title),dpi=100)
    plt.close()

    
def writeResult(filename,folder,params,run):
    
    with open("/vol/bitbucket/ttc14/thesis/result/{0}/{1}.txt".format(folder,filename),"a+") as f:
        if(run==0):
            f.write("no_of_episodes gamma learning_rate discount_noise punish ith_run avg_score actual_avg\n")
        for item in params:
            f.write(str(item))
            f.write(" ")
        f.write("\n")


def playGame(Q,env,num_episode,game_step):

    goals = []
    for episode in range(num_episode):
        state = env.reset()
        count = 0
        while count <= game_step:
            action = np.argmax(Q[state,:])
            new_state, reward, done, info = env.step(action)
            count+=1
            state = new_state
            if(done == True):
                break
        goals.append(reward)
    return goals
    
        

def main():
    game = "FrozenLakeNotSlippery-v0"
    env = gym.make(game)

    ###### Random Action Sampling ########
    '''
    num_epi = 1
    game_cnt = 100
    run_cnt = 1
    
    for i in range(run_cnt):
        rewards = randomSampling(env,num_epi,game_cnt)
        avg_rewards = sum(rewards)/num_epi
        print(avg_rewards)
    '''
    ############ Q-Learning ###############
    
    num_episode = 180
    game_step = 100
    gamma  = 0.9
    learning_rate = 0.8
    discount_noise = True                   # False to not use any Gaussian noise on action selection
    diminishing_weight = True              # False to not use the discounted weight for noise in late episodes
    punish = False                          # True to set a punishment for stepping into the hole in FrozenLake-v0
    punish_val = -1

    run = 5                                 # Number of runs to train the agent 
    save = False                            # True to save the picture generated from evalEpisode()
    folder = "frozen_lake_not_slippery"
    
    for i in range(run):
        
        (goals, Q) = qLearning(env,num_episode,gamma,learning_rate,game_step,punish,discount_noise,punish_val,diminishing_weight)
        avg_score = sum(goals)/num_episode
        print("Average score per episode:", avg_score)

        ############ Using Final Q Table to Play Games without further Update ##################
        # '''
        actual_goals = playGame(Q,env,num_episode,game_step)
        actual_avg = sum(actual_goals)/num_episode
        print("Average actual score per episode:", actual_avg)
        # '''

        ############## Store the Result ###############
        
        filename = "Tabular_QLearning_Result_of_{0}_{1}_episodes".format(game,num_episode) 
        params = [num_episode,gamma,learning_rate,discount_noise,punish,i,avg_score,actual_avg]
        
        # '''
        writeResult(filename,folder,params,i)
        # '''
        
        ############# Plot the Change of Goal against Episode ####################
        '''    
        title = "{0}th_Tabular_QLearning_Result_of_{1}_{2}_episodes".format(i,game,num_episode)
        evalEpisode(goals,num_episode,title,save,folder)
        
        '''
        
        
if __name__ == "__main__":
    main()
