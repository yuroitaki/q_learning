import numpy as np

class Tabular_Q_Agent:

    def __init__(self,gamma,learning_rate,obs_n,act_n,max_epsilon,
                 min_epsilon,discount_noise,diminishing_weight,max_epi):

        self.max_epi = max_epi
        self.obs_n = obs_n
        self.act_n = act_n
        
        self.Q = np.zeros([obs_n,act_n])
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.discount_noise = discount_noise
        self.diminishing_weight = diminishing_weight

        self.visit_count = np.zeros([obs_n,act_n])
        self.reward = [[[] for action in range(self.act_n)] for state in range(self.obs_n)]

        
    def act(self,state,episode):

        if self.discount_noise == True and self.diminishing_weight == True:
            action = np.argmax(self.Q[state,:]+np.random.randn(1,self.act_n)*(1/(episode+1)))
        elif self.discount_noise == True and self.diminishing_weight == False:
            action = np.argmax(self.Q[state,:]+np.random.randn(1,self.act_n))
        elif self.discount_noise == "epsilon":
            action = self.epsilonGreedy(state,episode)
        else:
            action = self.play(state)

        self.visit_count[state,action] += 1
        return action    
        

    def train(self,new_state,reward,state,action):
        
        optimal_Q = np.max(self.Q[new_state,:])
        td_delta = reward + self.gamma*(optimal_Q) - self.Q[state,action]
        self.Q[state,action] += self.learning_rate*(td_delta)
        
        return td_delta


    def count_train(self,new_state,reward,state,action,const):

        exploration_bonus  = const / np.sqrt(self.visit_count[state,action])
        optimal_Q = np.max(self.Q[new_state,:])
        td_delta = reward + exploration_bonus + self.gamma*(optimal_Q) - self.Q[state,action]
        self.Q[state,action] += self.learning_rate*(td_delta)
        
        return td_delta

    
    def model_train(self,new_state,reward,state,action,const):

        exploration_bonus  = const / np.sqrt(self.visit_count[state,action])
        mean_reward = self.calcAvgReward(reward,state,action)
        optimal_Q = np.max(self.Q[new_state,:])
        td_delta = mean_reward + exploration_bonus + self.gamma*(optimal_Q) - self.Q[state,action]
        # td_delta = mean_reward + self.gamma*(optimal_Q) - self.Q[state,action]
        self.Q[state,action] += self.learning_rate*(td_delta)

        return td_delta

    
    def calcAvgReward(self,im_reward,state,action):
        
        self.reward[state][action].append(im_reward)
        mean_r = sum(self.reward[state][action])/self.visit_count[state,action]
        return mean_r

    
    def epsilonGreedy(self,state,episode):

        ########## Linear Decay ###############
        '''
        use_epsilon = -(episode/self.max_epi) + self.max_epsilon
        if(use_epsilon < self.min_epsilon):
        use_epsilon = self.min_epsilon
        ''' 
        ########## Exponential Decay ##########
        # '''
        use_epsilon = self.max_epsilon * (1/(episode+1))
        # '''
        
        rand_num = np.random.uniform(0,1) *(1/(episode+1))
        if(use_epsilon > rand_num):
            action = np.random.random_integers(0,self.act_n-1)
        else:
            action = np.argmax(self.Q[state,:])
        
        return action

    
    def play(self,state):

        action = np.argmax(self.Q[state,:])
        return action
        
