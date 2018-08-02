import numpy as np

class Tabular_Q_Agent:

    def __init__(self,discount_factor,obs_n,act_n,max_epsilon,min_epsilon,exp_strategy,
                 diminishing,max_epi,q_update,epsilon_type,epsilon_rate,epsilon_const,update_policy):

        self.max_epi = max_epi
        self.obs_n = obs_n
        self.act_n = act_n
        
        self.q_update = q_update
        self.exp_strategy = exp_strategy
        self.discount_factor = discount_factor
        self.update_policy = update_policy
        
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_type = epsilon_type
        self.epsilon_rate = epsilon_rate
        self.epsilon_const = epsilon_const
        self.diminishing = diminishing

        self.Q = np.zeros([obs_n,act_n+1])
        self.visit_count = np.zeros([obs_n,act_n+1])
        self.M = np.zeros([obs_n,act_n+1])
        self.U = np.zeros([obs_n,act_n+1])
        self.var = np.zeros([obs_n,act_n+1]) 

        self.monte_goal = np.zeros([obs_n,act_n+1])
        self.monte_var = np.zeros([obs_n,act_n+1])
                
        self.initialiseTable()

        
    def initialiseTable(self):

        for row in range(self.obs_n):
            self.Q[row][self.act_n] = row
            self.M[row][self.act_n] = row
            self.U[row][self.act_n] = row
            self.visit_count[row][self.act_n] = row
            self.var[row][self.act_n] = row
            self.monte_goal[row][self.act_n] = row
            self.monte_var[row][self.act_n] = row

            
    def act(self,state,episode):
        
        if self.q_update == "risk":
            table = self.U
        else:
            table = self.Q
        
        if self.exp_strategy == "softmax" and self.diminishing == True:
            action = np.argmax(table[state,:-1]+np.random.randn(1,self.act_n)*(1/(episode+1)))
        elif self.exp_strategy == "softmax" and self.diminishing == False:
            action = np.argmax(table[state,:-1]+np.random.randn(1,self.act_n))
            
        elif self.exp_strategy == "epsilon":
            action = self.epsilonGreedy(state,episode)
            
        elif self.exp_strategy == "greedy":
            action = self.optimalAction(table,state)

        self.visit_count[state,action] += 1
        return action    


    ########## Normal Exploration Bonus ##################
    
    def train(self,new_state,reward,state,action,learning_rate,exploration_bonus=0):
        
        optimal_Q = np.max(self.Q[new_state,:-1])
        td_delta = reward + exploration_bonus + self.discount_factor*(optimal_Q) - self.Q[state,action]
        self.Q[state,action] += learning_rate*(td_delta)
        
        return td_delta

    ########### Count-Based Exploration Bonus ###########

    
    def count_train(self,new_state,reward,state,action,learning_rate,beta):

        exploration_bonus  = beta/np.sqrt(self.visit_count[state,action])
        
        return self.train(new_state,reward,state,action,learning_rate,exploration_bonus)

    
    ####### Risk Seeking Exploration ###########
    
    def risk_train(self,new_state,reward,state,action,risk_level,epi,learning_rate):
        
        # new_action = self.play(new_state,epi)
        
        if self.update_policy == "greedy":
            new_action = self.optimalAction(self.U,new_state)
            
        elif self.update_policy == "epsilon":
             new_action = self.epsilonGreedy(new_state,epi)
        
        optimal_Q = self.Q[new_state,new_action]
        delta_Q = reward + self.discount_factor*(optimal_Q) - self.Q[state,action]
        self.Q[state,action] += learning_rate*(delta_Q)

        optimal_M = self.M[new_state,new_action]
        delta_M = (reward**2) + (2*self.discount_factor*reward*(optimal_Q)) + ((self.discount_factor**2)*optimal_M) - self.M[state,action]
        self.M[state,action] += learning_rate*(delta_M)

        variance = self.M[state,action] - (self.Q[state,action]**2)
        # if(variance < 0):
        #     print(variance)
        self.var[state,action] = variance
        self.U[state,action] = self.Q[state,action] + risk_level*(max(0,variance))
            

    ########### Epsilon Greedy ################
    
    def epsilonGreedy(self,state,episode):

        if self.q_update == "risk":
            table = self.U
        else:
            table = self.Q
        
        if self.epsilon_type == "linear":
            use_epsilon = -(episode/self.max_epi) + self.max_epsilon
            if(use_epsilon < self.min_epsilon):
                use_epsilon = self.min_epsilon
                
        elif self.epsilon_type == "exponential":
            use_epsilon = self.max_epsilon * (1/(episode+1)**self.epsilon_rate)

        elif self.epsilon_type == "constant":
            use_epsilon = self.epsilon_const
            
        rand_num = np.random.uniform(0,1) 
        if(use_epsilon > rand_num):
            action = np.random.random_integers(0,self.act_n-1)
        else:
            action = self.optimalAction(table,state)
        
        return action


    def play(self,state,episode):
 
        if self.q_update == "risk":
            table = self.U
        else:
            table = self.Q

        # if self.update_policy == "greedy":
        action = self.optimalAction(table,state)
            
        # elif self.update_policy == "epsilon":
        #     action = self.epsilonGreedy(state,episode)
        
        return action
        

    def optimalAction(self,table,state):

        max_Q = max(table[state,:-1])
        index = 0
        index_list = []
        
        for a in table[state,:-1]:
            if max_Q == a:
                index_list.append(index)
            index += 1

        rand = np.random.choice(index_list)
        return rand


    def learningRate(self,episode,power=0.85):

        # alpha = 1/((episode+1)**power)
        alpha = 0.7

        return alpha
            
    


            
