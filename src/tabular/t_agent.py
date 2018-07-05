import numpy as np

class Tabular_Q_Agent:

    def __init__(self,gamma,obs_n,act_n,max_epsilon,min_epsilon,discount_noise,
                 diminishing_weight,max_epi,explore,esp_rate):

        self.max_epi = max_epi
        self.obs_n = obs_n
        self.act_n = act_n
        
        self.Q = np.zeros([obs_n,act_n])
        self.gamma = gamma
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.explore = explore
        self.esp_rate = esp_rate
        
        self.discount_noise = discount_noise
        self.diminishing_weight = diminishing_weight
        

        self.visit_count = np.zeros([obs_n,act_n])
        self.M = np.zeros([obs_n,act_n])
        self.U = np.zeros([obs_n,act_n])
        
        #self.reward = [[[] for action in range(self.act_n)] for state in range(self.obs_n)]

        
    def act(self,state,episode,rate):

        if self.discount_noise == True and self.diminishing_weight == True:
            action = np.argmax(self.Q[state,:]+np.random.randn(1,self.act_n)*(1/(episode+1)))
        elif self.discount_noise == True and self.diminishing_weight == False:
            action = np.argmax(self.Q[state,:]+np.random.randn(1,self.act_n))
            
        elif self.discount_noise == "epsilon":
            action = self.epsilonGreedy(state,episode,rate)
            
        elif self.discount_noise == "risk":
            action = self.optimalAction(self.U,state)
            
        else:
            action = self.optimalAction(self.Q,state)          # greedy method

        self.visit_count[state,action] += 1
        return action    
        

    ########## Normal Exploration Bonus ##################
    
    def train(self,new_state,reward,state,action,learning_rate,exploration_bonus=0):
        
        optimal_Q = np.max(self.Q[new_state,:])
        td_delta = reward + exploration_bonus + self.gamma*(optimal_Q) - self.Q[state,action]
        self.Q[state,action] += learning_rate*(td_delta)
        
        return td_delta

    ########### Count-Based Exploration Bonus ###########

    
    def count_train(self,new_state,reward,state,action,learning_rate,beta):

        exploration_bonus  = beta/np.sqrt(self.visit_count[state,action])
        
        return self.train(new_state,reward,state,action,learning_rate,exploration_bonus)

    
    ####### Risk Seeking Exploration ###########
    
    def risk_train(self,new_state,reward,state,action,risk_level,epi,rate,learning_rate):
        
        new_action = self.optimalAction(self.U,new_state)
        # new_action = self.epsilonGreedy(new_state,epi,rate)
        
        optimal_Q = self.Q[new_state,new_action]
        delta_Q = reward + self.gamma*(optimal_Q) - self.Q[state,action]
        self.Q[state,action] += learning_rate*(delta_Q)

        optimal_M = self.M[new_state,new_action]
        delta_M = (reward**2) + (2*self.gamma*reward*(optimal_Q)) + ((self.gamma**2)*optimal_M) - self.M[state,action]
        self.M[state,action] += learning_rate*(delta_M)

        # self.U[state,action] = self.Q[state,action] + risk_level*(max(0,self.M[state,action] - (self.Q[state,action]**2)))

        delta_U = self.Q[state,action] + risk_level*(max(0,self.M[state,action] - (self.Q[state,action]**2)))
        self.U[state,action] += learning_rate*(delta_U)

        return delta_Q, delta_M
    

    ########### Epsilon Greedy ################
    
    def epsilonGreedy(self,state,episode,rate):

        # if self.explore == "risk":
        #     table = self.U
        # else:
        #     table = self.Q

        table = self.Q
        
        if rate == "linear":
            use_epsilon = -(episode/self.max_epi) + self.max_epsilon
            if(use_epsilon < self.min_epsilon):
                use_epsilon = self.min_epsilon
                
        elif rate == "exponential":
            use_epsilon = self.max_epsilon * (1/(episode+1)**self.esp_rate)

        rand_num = np.random.uniform(0,1) #*(1/(episode+1))
        if(use_epsilon > rand_num):
            action = np.random.random_integers(0,self.act_n-1)
        else:
            action = self.optimalAction(table,state)
        
        return action


    def play(self,state):

        action = self.optimalAction(self.Q,state)
        return action
        

    def optimalAction(self,table,state):

        max_Q = max(table[state,:])
        index = 0
        index_list = []
        
        for a in table[state,:]:
            if max_Q == a:
                index_list.append(index)
            index += 1

        rand = np.random.choice(index_list)
        return rand


    def learningRate(self,episode,power=0.85):

        alpha = 1/((episode+1)**power)
        # alpha = 0.8

        return alpha
            
    
    ####### Model-Based Training with only mean reward, no mean transition, not functioning ########
    
    def model_train(self,new_state,reward,state,action,beta):

        exploration_bonus  = beta / np.sqrt(self.visit_count[state,action])
        mean_reward = self.calcAvgReward(reward,state,action)
        optimal_Q = np.max(self.Q[new_state,:])
        td_delta = mean_reward + exploration_bonus + self.gamma*(optimal_Q) - self.Q[state,action]
        self.Q[state,action] += self.learning_rate*(td_delta)

        return td_delta

    
    def calcAvgReward(self,im_reward,state,action):
        
        self.reward[state][action].append(im_reward)
        mean_r = sum(self.reward[state][action])/self.visit_count[state,action]
        return mean_r


            
