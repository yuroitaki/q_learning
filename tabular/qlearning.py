import gym
import numpy as np
import matplotlib.pyplot as plt

def randomSampling(env):
    
    count  = 0
    reward = 0
    while reward < 1:
        env.reset()
        done = False
        while done != True:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            # env.render()
        # if(reward < 1):
        #     print("**************")
        #     print("Game over boi!")
        #     print("**************")
        count += 1
    # print("**************")
    # print("You win at ",count," boi!")
    # print("**************")
    print(count)
    

def qLearning(env,num_episode,gamma,learning_rate,game_step,punish,discount_noise,punish_val):
    
    Q = np.zeros([env.observation_space.n,env.action_space.n])
    goals = []
    counter  = 0
    for episode in range(num_episode):
        state = env.reset()
        acc_reward = 0
        count = 0
        
        while count <= game_step:
            if discount_noise == True:
                action = np.argmax(Q[state,:]+np.random.randn(1,env.action_space.n)*(1/(episode+1)))
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
    return goals


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
            f.write("no_of_episodes gamma learning_rate discount_noise punish ith_run avg_score\n")
        for item in params:
            f.write(str(item))
            f.write(" ")
        f.write("\n")
        

    

def main():
    game = "FrozenLake-v0"
    env = gym.make(game)
    # randomSampling(env)

    ############ Q-Learning ###############
    num_episode = 100000
    game_step = 100
    gamma  = 0.9
    learning_rate = 0.8
    discount_noise = True
    punish = False
    punish_val = -1

    run = 5
    save = False
    folder = "frozen_lake"
    
    for i in range(run):
        goals = qLearning(env,num_episode,gamma,learning_rate,game_step,punish,discount_noise,punish_val)
        title = "{0}th_Tabular_QLearning_Result_of_{1}_{2}_episodes".format(i,game,num_episode)
        # evalEpisode(goals,num_episode,title,save,folder)
        filename = "Tabular_QLearning_Result_of_{0}_{1}_episodes".format(game,num_episode) 
        avg_score = sum(goals)/num_episode
        params = [num_episode,gamma,learning_rate,discount_noise,punish,i,avg_score]
        writeResult(filename,folder,params,i)
        print("Average score per episode:", avg_score)
    
if __name__ == "__main__":
    main()
