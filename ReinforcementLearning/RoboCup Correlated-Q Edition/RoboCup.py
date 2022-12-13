import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from scipy.linalg import block_diag
import time
solvers.options['show_progress'] = False

class Robocup:
    def __init__(self):
        self.position = [np.array([0, 2]), np.array([0, 1])]
        self.goal = [0, 3]
        self.ball = 1
        NORTH = [-1, 0]
        SOUTH = [1, 0]
        EAST = [0, 1]
        WEST = [0, -1]
        STICK = [0, 0]
        self.game_actions = [NORTH, EAST, SOUTH, WEST, STICK]    
        
    def __collide(self,player1,player2):
        return True if np.array_equal(player1,player2) else False
    def __loose_ball(self,player1,player2):
        if (self.ball == player1) : self.ball = player2
    def __move_ball(self,player_pos):
        return True if  0<=player_pos[0]<=1 and  0<=player_pos[1]<=3 else False
    def __init_reward_state(self):
        return [self.position[0][0] * 4 + self.position[0][1], self.position[1][0] * 4 + self.position[1][1], self.ball], np.array([0, 0]),False
    def __play(self,player1,player2,actions,new_pos):
        new_pos[player1] = self.position[player1] + self.game_actions[actions[player1]]
        collision = self.__collide(new_pos[player1],self.position[player2])
        if collision:
            self.__loose_ball(player1,player2)
        elif self.__move_ball(new_pos[player1]):
            self.position[player1] = new_pos[player1]
            if self.ball == player1 and self.position[player1][1] == self.goal[player1]:
                return new_pos,[self.position[0][0] * 4 + self.position[0][1], self.position[1][0] * 4 + self.position[1][1], self.ball], np.array([100, -100]) * [1, -1][player2] , True
            elif self.ball == player1 and self.position[player1][1] == self.goal[player2]:
                return new_pos,[self.position[0][0] * 4 + self.position[0][1], self.position[1][0] * 4 + self.position[1][1], self.ball], np.array([-100, 100]) * [1, -1][player2] , True
        return new_pos,[self.position[0][0] * 4 + self.position[0][1], self.position[1][0] * 4 + self.position[1][1], self.ball], np.array([0, 0]),False
                    
    def move(self, actions):
        player_A = np.random.choice([0, 1], 1)[0]
        player_B = 1-player_A
        new_pos = self.position.copy()
        next_state, reward, done = self.__init_reward_state()
        new_pos,next_state, reward, done = self.__play(player_A,player_B,actions,new_pos)
        if done==False :
            new_pos,next_state, reward, done = self.__play(player_B,player_A,actions,new_pos)
        return next_state,reward, done

## Q
class QLearner:
    
    def __init__(self, observation_space=8,action_space=8, \
                 players=2,number_of_actions=5 ,\
                 episodes=1000000,gamma=0.9,alpha=0.01,\
                 epsilon_begin=0.1,epsilon_end=0) :
        self.episodes = episodes
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon_begin = epsilon_begin
        self.epsilon_end = epsilon_end
        self.epsilon_periods = episodes/2
        self.observation_space = observation_space
        self.action_space = action_space
        self.players = players
        self.number_of_actions = number_of_actions

    def __choose_Q_action(self,Q_table, state, epsilon):
        if np.random.random() > epsilon:
            return np.random.choice(np.where(Q_table[state[0]][state[1]][state[2]] == max(Q_table[state[0]][state[1]][state[2]]))[0], 1)[0]
        else:
            return np.random.choice([0,1,2,3,4], 1)[0]
    def __actions_Q(self,Q_tables,state,epsilon):
        return [self.__choose_Q_action(Q_tables[0], state, epsilon), self.__choose_Q_action(Q_tables[1], state,epsilon)]

    def __epsilon_Q(self,epsilon_begin,epsilon_end,epsilon_periods,episode):
        epsilon = epsilon_end + (epsilon_begin - epsilon_end) * np.exp(-1.0 * episode / epsilon_periods)
        return epsilon
    def __Q_tables(self,observation_space, action_space, players, number_of_actions):
        return np.zeros((observation_space, action_space, players, number_of_actions)), np.zeros((observation_space, action_space, players, number_of_actions))

    def __alpha(self,episode,episodes,alpha):
        alpha = 1 / (episode / alpha / episodes + 1)
        return alpha

    def __Q_update(self,Q_table,state,actions,state_next,rewards,done,alpha,gamma,player_action,reward_idx):
        if done == False:
            return Q_table[state[0]][state[1]][state[2]][actions[player_action[0]]] + alpha * (rewards[reward_idx] + gamma * max(Q_table[state_next[0]][state_next[1]][state_next[2]]) - Q_table[state[0]][state[1]][state[2]][actions[player_action[0]]])
        elif done == True:
            return Q_table[state[0]][state[1]][state[2]][actions[player_action[0]]] + alpha * (rewards[reward_idx] - Q_table[state[0]][state[1]][state[2]][actions[player_action[0]]])
    def train(self):
        np.random.seed(5)
        Q_A_table, Q_B_table = self.__Q_tables(self.observation_space, self.action_space, self.players, self.number_of_actions)
        errors = []
        begin_time = time.time()
        episode = 0
        while episode < self.episodes:
            robocup = Robocup()
            done = False
            state = [robocup.position[0][0] * 4 + robocup.position[0][1], robocup.position[1][0] * 4 + robocup.position[1][1], robocup.ball]
            while not done:
                if (episode + 1) % 100000 == 0: print(episode + 1, ', ', np.round(time.time() - begin_time, 0), 'seconds')
                episode += 1
                actions = self.__actions_Q([Q_A_table, Q_B_table],state,self.__epsilon_Q(self.epsilon_begin,self.epsilon_end,self.epsilon_periods,episode))
                state_next, rewards, done = robocup.move(actions)
                q_t_1 = Q_A_table[2][1][1][2]
                Q_A_table[state[0]][state[1]][state[2]][actions[0]] = self.__Q_update(Q_A_table,state,actions,state_next,rewards,done,self.__alpha(episode,self.episodes,self.alpha),self.gamma,[0],0)
                Q_B_table[state[0]][state[1]][state[2]][actions[1]] = self.__Q_update(Q_B_table,state,actions,state_next,rewards,done,self.__alpha(episode,self.episodes,self.alpha),self.gamma,[1],1)
                qt = Q_A_table[2][1][1][2]
                state = state_next
                errors.append(np.abs(qt - q_t_1))
        return np.array(errors)[np.where(np.array(errors) > 0)]

## Friend-Q
class FriendQLearner:
    
    def __init__(self, observation_space=8,action_space=8, \
                 players=2,number_of_actions=5 ,\
                 episodes=1000000,gamma=0.9,alpha=0.01,\
                 epsilon_begin=0.1,epsilon_end=0) :
        self.episodes = episodes
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon_begin = epsilon_begin
        self.epsilon_end = epsilon_end
        self.epsilon_periods = episodes/2
        self.observation_space = observation_space
        self.action_space = action_space
        self.players = players
        self.number_of_actions = number_of_actions

    def __choose_Q_action(self,Q_table, state, epsilon):
        if np.random.random() > epsilon:
            max_idx = np.where(Q_table[state[0]][state[1]][state[2]] == np.max(Q_table[state[0]][state[1]][state[2]]))
            return max_idx[1][np.random.choice(range(len(max_idx[0])), 1)[0]]
        else:
            return np.random.choice([0,1,2,3,4], 1)[0]
    def __actions_Q(self,Q_tables,state,epsilon):
        return [self.__choose_Q_action(Q_tables[0], state, epsilon), self.__choose_Q_action(Q_tables[1], state,epsilon)]

    def __epsilon_Q(self,epsilon_begin,epsilon_end,epsilon_periods,episode):
        epsilon = epsilon_end + (epsilon_begin - epsilon_end) * np.exp(-1.0 * episode / epsilon_periods)
        return epsilon
    def __Q_tables(self,observation_space, action_space, players, number_of_actions):
        return np.zeros((observation_space, action_space, players, number_of_actions,number_of_actions)) , np.zeros((observation_space, action_space, players, number_of_actions,number_of_actions))

    def __alpha(self,episode,episodes,alpha):
        alpha = 1 / (episode / alpha / episodes + 1)
        return alpha

    def __Q_update(self,Q_table,state,actions,state_next,rewards,done,alpha,gamma,player_action,reward_idx):
        if done == False:
            return Q_table[state[0]][state[1]][state[2]][actions[player_action[0]]][actions[player_action[1]]] + alpha * (rewards[reward_idx] + gamma * np.max(Q_table[state_next[0]][state_next[1]][state_next[2]]) - Q_table[state[0]][state[1]][state[2]][actions[player_action[0]]][actions[player_action[1]]])
        elif done ==True:
            return Q_table[state[0]][state[1]][state[2]][actions[player_action[0]]][actions[player_action[1]]] + alpha * (rewards[reward_idx] - Q_table[state[0]][state[1]][state[2]][actions[player_action[0]]][actions[player_action[1]]])
    def train(self):
        np.random.seed(5)
        Q_A_table, Q_B_table = self.__Q_tables(self.observation_space, self.action_space, self.players, self.number_of_actions)
        errors = []
        begin_time = time.time()
        episode = 0
        while episode < self.episodes:
            robocup = Robocup()
            done = False
            state = [robocup.position[0][0] * 4 + robocup.position[0][1], robocup.position[1][0] * 4 + robocup.position[1][1], robocup.ball]
            while not done:
                if (episode + 1) % 100000 == 0: print(episode + 1, ', ', np.round(time.time() - begin_time, 0), 'seconds')
                episode += 1
                actions = self.__actions_Q([Q_A_table, Q_B_table],state,self.__epsilon_Q(self.epsilon_begin,self.epsilon_end,self.epsilon_periods,episode))
                state_next, rewards, done = robocup.move(actions)
                q_t_1 = Q_A_table[2][1][1][4][2]
                Q_A_table[state[0]][state[1]][state[2]][actions[1]][actions[0]] = self.__Q_update(Q_A_table,state,actions,state_next,rewards,done,self.__alpha(episode,self.episodes,self.alpha),self.gamma,[1,0],0)
                Q_A_table[state[0]][state[1]][state[2]][actions[0]][actions[1]] = self.__Q_update(Q_A_table,state,actions,state_next,rewards,done,self.__alpha(episode,self.episodes,self.alpha),self.gamma,[0,1],1)
                qt = Q_A_table[2][1][1][4][2]
                state = state_next
                errors.append(np.abs(qt - q_t_1))
        return np.array(errors)[np.where(np.array(errors) > 0)]

## Foe-Q
class FoeQLearner:
    
    def __init__(self, observation_space=8,action_space=8, \
                 players=2,number_of_actions=5 ,\
                 episodes=1000000,gamma=0.9,alpha=0.01,\
                 epsilon_begin=0.1,epsilon_end=0) :
        self.episodes = episodes
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = 10**(np.log10(alpha)/episodes)
        self.epsilon_begin = epsilon_begin
        self.epsilon_end = epsilon_end
        self.epsilon_periods = episodes/2
        self.epsilon_decay = 10**(np.log10(epsilon_end)/episodes)
        self.observation_space = observation_space
        self.action_space = action_space
        self.players = players
        self.number_of_actions = number_of_actions

    def __choose_Pi_action(self,Pi_table, state, epsilon):
        if np.random.random() > epsilon:
            return np.random.choice([0,1,2,3,4], 1, p=Pi_table[state[0]][state[1]][state[2]])[0]
        else:
            return np.random.choice([0,1,2,3,4], 1)[0]
    def __actions_Pi(self,Pi_tables,state,epsilon):
        return [self.__choose_Pi_action(Pi_tables[0], state, epsilon), self.__choose_Pi_action(Pi_tables[1], state,epsilon)]
    def __epsilon_Pi(self,epsilon_decay,episode):
        return epsilon_decay ** episode

    def __alpha(self,alpha_decay,episode):
       return alpha_decay ** episode

    def __Q_tables(self,observation_space, action_space, players, number_of_actions):
        return np.zeros((observation_space, action_space, players, number_of_actions,number_of_actions)) * 1.0 , np.zeros((observation_space, action_space, players, number_of_actions,number_of_actions)) * 1.0

    def __V_tables(self,observation_space, action_space, players, number_of_actions):
        return np.ones((observation_space, action_space, players)) * 1.0, np.ones((observation_space, action_space, players)) * 1.0

    def __Pi_tables(self,observation_space, action_space, players, number_of_actions):
        return np.ones((observation_space, action_space, players, number_of_actions)) * 1/number_of_actions , np.ones((observation_space, action_space, players, number_of_actions)) * 1/number_of_actions

    def __Q_update(self,Q_table,V_table,state,actions,state_next,rewards,done,alpha,gamma,player_action,reward_idx):
        return (1 - alpha) * Q_table[state[0]][state[1]][state[2]][actions[player_action[0]]][actions[player_action[1]]] + alpha * (rewards[reward_idx] + gamma * V_table[state_next[0]][state_next[1]][state_next[2]])

    def __solve(self,Q_table, state):
        c = matrix([0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        G = matrix(np.append(np.append(-Q_table[state[0]][state[1]][state[2]], np.ones((self.number_of_actions,1)), axis=1), np.append(-np.eye(self.number_of_actions), np.zeros((self.number_of_actions,1)), axis=1), axis=0))
        h = matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        A = matrix([[1.0], [1.0], [1.0], [1.0], [1.0], [0.0]])
        b = matrix(1.0)
        Solution = solvers.lp(c, G, h, A, b)
        return np.abs(Solution['x'][0:5]).reshape((5,)) / sum(np.abs(Solution['x'][0:5])), np.array(Solution['x'][5])


    def train(self):
        np.random.seed(100)
        Q_A_table, Q_B_table = self.__Q_tables(self.observation_space, self.action_space, self.players, self.number_of_actions)
        V_A_table, V_B_table = self.__V_tables(self.observation_space, self.action_space, self.players, self.number_of_actions)
        Pi_A_table, Pi_B_table = self.__Pi_tables(self.observation_space, self.action_space, self.players, self.number_of_actions)
        errors = []
        begin_time = time.time()
        episode = 0
        while episode < self.episodes:
            robocup = Robocup()
            done = False
            state = [robocup.position[0][0] * 4 + robocup.position[0][1], robocup.position[1][0] * 4 + robocup.position[1][1], robocup.ball]
            while not done:
                if (episode + 1) % 100000 == 0: print(episode + 1, ', ', np.round(time.time() - begin_time, 0), 'seconds')
                episode += 1
                actions = self.__actions_Pi( [Pi_A_table, Pi_B_table],state,self.__epsilon_Pi(self.epsilon_decay,episode) )
                state_next, rewards, done = robocup.move(actions)
                q_t_1 = Q_A_table[2][1][1][4][2]
                Q_A_table[state[0]][state[1]][state[2]][actions[1]][actions[0]] = self.__Q_update(Q_A_table,V_A_table,state,actions,state_next,rewards,done,self.__alpha(self.alpha_decay,episode),self.gamma,[1,0],0)
                Pi_A_table[state[0]][state[1]][state[2]] , V_A_table[state[0]][state[1]][state[2]] = self.__solve(Q_A_table, state)
                Q_B_table[state[0]][state[1]][state[2]][actions[1]][actions[0]] = self.__Q_update(Q_B_table,V_B_table,state,actions,state_next,rewards,done,self.__alpha(self.alpha_decay,episode),self.gamma,[0,1],1)
                Pi_B_table[state[0]][state[1]][state[2]] , V_B_table[state[0]][state[1]][state[2]] = self.__solve(Q_B_table, state)
                q_t = Q_A_table[2][1][1][4][2]
                state = state_next
                errors.append(np.abs(q_t - q_t_1))
        return np.array(errors)[np.where(np.array(errors) > 0)]

## Correlated-Q
class CorrelatedQLearner:
    
    def __init__(self, observation_space=8,action_space=8, \
                 players=2,number_of_actions=5 ,\
                 episodes=1000000,gamma=0.9,alpha=0.01,\
                 epsilon_begin=0.1,epsilon_end=0) :
        self.episodes = episodes
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = 10**(np.log10(alpha)/episodes)
        self.epsilon_begin = epsilon_begin
        self.epsilon_end = epsilon_end
        self.epsilon_periods = episodes/2
        self.epsilon_decay = 10**(np.log10(epsilon_end)/episodes)
        self.observation_space = observation_space
        self.action_space = action_space
        self.players = players
        self.number_of_actions = number_of_actions

    def __choose_Pi_action(self,Pi_table, state, epsilon):
        if np.random.random() > epsilon:
            idx = np.random.choice(np.arange(25), 1, p=pi[state[0]][state[1]][state[2]].reshape(25))
            return np.array([idx // 5, idx % 5]).reshape(2)
        else:
            idx = np.random.choice(np.arange(25), 1)
            return np.array([idx // 5, idx % 5]).reshape(2)

    def __actions_Pi(self,Pi_tables,state,epsilon):
        return self.__choose_Pi_action(Pi_tables, state, epsilon)
    def __epsilon_Pi(self,epsilon_decay,episode):
        return epsilon_decay ** episode

    def __alpha(self,alpha_decay,episode):
        alpha = alpha_decay ** episode
        return alpha
    
    def __Q_tables(self,observation_space, action_space, players, number_of_actions):
        return np.zeros((observation_space, action_space, players, number_of_actions,number_of_actions)) * 1.0 , np.zeros((observation_space, action_space, players, number_of_actions,number_of_actions)) * 1.0

    def __V_tables(self,observation_space, action_space, players, number_of_actions):
        return np.ones((observation_space, action_space, players)) * 1.0, np.ones((observation_space, action_space, players)) * 1.0

    def __Pi_tables(self,observation_space, action_space, players, number_of_actions):
        return np.ones((observation_space, action_space, players, number_of_actions)) * 1/(number_of_actions**2) 

    def __Q_update(self,Q_table,V_table,state,actions,state_next,rewards,done,alpha,gamma,player_action,reward_idx):
        return (1 - alpha) * Q_table[state[0]][state[1]][state[2]][actions[player_action[0]]][actions[player_action[0]]] + alpha * (rewards[reward_idx] + gamma * V_table[state_next[0]][state_next[1]][state_next[2]])

    def __G_Matrix_Condition(self,Q_table):
        r_ix = (1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23)
        c_ix = (0, 5, 10, 15, 20, 1, 6, 11, 16, 21, 2, 7, 12, 17, 22, 3, 8, 13, 18, 23, 4, 9, 14, 19, 24)
        Qs1 = Q_table[0][state[0]][state[1]][state[2]]
        s1 = block_diag(Qs - Qs1[0, :], Qs1 - Qs1[1, :], Qs1 - Qs1[2, :], Qs1 - Qs1[3, :], Qs1 - Qs1[4, :])
        Qs2 = Q_table[1][state[0]][state[1]][state[2]]
        s2 = block_diag(Qs2 - Qs2[0, :], Qs2 - Qs2[1, :], Qs2 - Qs2[2, :], Qs2 - Qs2[3, :], Qs2 - Qs2[4, :])
        return np.append(s1[r_ix, :], s2[r_ix, :][:, c_ix], axis=0)


    def __solve(self,Q_table, state):
        c = matrix((Q_table[0][state[0]][state[1]][state[2]] + Q_table[1][state[0]][state[1]][state[2]].T).reshape(self.number_of_actions**2))
        G = matrix(np.append(self.__G_Matrix_Condition(Q_table), -np.eye(self.number_of_actions**2), axis=0))
        h = matrix(np.zeros((self.action_space+self.number_of_actions) * self.number_of_actions,dtype=float) )
        A = matrix(np.ones((1, self.number_of_actions**2)))
        b = matrix(1.0)
        Solution = solvers.lp(c, G, h, A, b)
        pi,v1,v2 = None,None,None
        if Solution['x'] is not None:
            pi,v1,v2 = np.abs(np.array(Solution['x']).reshape((5, 5))) / sum(np.abs(Solution['x'])) , np.sum(prob * Q1[state[0]][state[1]][state[2]]) , np.sum(prob * Q2[state[0]][state[1]][state[2]].T) 
        return pi,v1,v2

    def train(self):
        np.random.seed(100)
        Q_A_table, Q_B_table = self.__Q_tables(self.observation_space, self.action_space, self.players, self.number_of_actions)
        V_A_table, V_B_table = self.__V_tables(self.observation_space, self.action_space, self.players, self.number_of_actions)
        Pi_table = self.__Pi_tables(self.observation_space, self.action_space, self.players, self.number_of_actions)
        errors = []
        begin_time = time.time()
        episode = 0
        while episode < self.episodes:
            robocup = Robocup()
            done = False
            state = [robocup.position[0][0] * 4 + robocup.position[0][1], robocup.position[1][0] * 4 + robocup.position[1][1], robocup.ball]
            while not done:
                if (episode + 1) % 100000 == 0: print(episode + 1, ', ', np.round(time.time() - begin_time, 0), 'seconds')
                episode += 1
                actions = self.__actions_Pi(Pi_A_table,state,self.__epsilon_Pi(self.epsilon_decay,episode) )
                state_next, rewards, done = robocup.move(actions)
                q_t_1 = Q_A_table[2][1][1][4][2]
                Q_A_table[state[0]][state[1]][state[2]][actions[0]][actions[1]] = self.__Q_update(Q_A_table,V_A_table,state,actions,state_next,rewards,done,self.__alpha(self.alpha_decay,episode),self.gamma,[0,1],0)
                Pi_A_table[state[0]][state[1]][state[2]] , V_A_table[state[0]][state[1]][state[2]] = self.__solve(Q_A_table, state)

                Q_B_table[state[0]][state[1]][state[2]][actions[1]][actions[0]] = self.__Q_update(Q_B_table,V_B_table,state,actions,state_next,rewards,done,self.__alpha(self.alpha_decay,episode),self.gamma,[0,1],1)
                Pi_B_table[state[0]][state[1]][state[2]] , V_B_table[state[0]][state[1]][state[2]] = self.__solve(Q_B_table, state)
                q_t = Q_A_table[2][1][1][4][2]
                state = state_next
                errors.append(np.abs(q_t - q_t_1))
        return np.array(errors)[np.where(np.array(errors) > 0)]



def plot(errors, title, filename):
    plt.figure(figsize=(10,6))
    plt.clf()
    plt.plot(errors, linestyle='-', linewidth=0.5)
    plt.title(title)
    plt.xlabel('Simulation Iterations')
    plt.ylabel('Q-value Difference')
    plt.ylim(0, 0.5)
    plt.savefig(filename)
    #plt.show()


if __name__=="__main__":


    print("### FriendQ-learning")
    q_learning = QLearner(alpha=0.001)
    q_learning_errors = q_learning.train()
    plot(q_learning_errors, 'Q-learning', "Q-learning")

    print("### FriendQ-learning")
    friend_q_learning = FriendQLearner(alpha=0.001)
    friend_q_learning_errors = friend_q_learning.train()
    plot(friend_q_learning_errors, 'FriendQ-learning', "FriendQ-learning")

    print("### FoeQ-learning")
    foe_q_learning = FoeQLearner(alpha=0.001,epsilon_end=0.001)
    foe_q_learning_errors = foe_q_learning.train()
    plot(foe_q_learning_errors, 'FoeQ-learning', "FoeQ-learning")

    print("### CorrelatedQ-learning")
    corr_q_learning = FoeQLearner(alpha=0.001,epsilon_end=0.001)
    corr_q_learning_errors = corr_q_learning.train()
    plot(corr_q_learning_errors, 'CorrelatedQ-learning', "CorrelatedQ-learning")


