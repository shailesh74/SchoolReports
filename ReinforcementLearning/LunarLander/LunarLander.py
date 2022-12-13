import gym, random
import numpy as np
from gym import wrappers
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import os
import matplotlib.pyplot as plt
#from keras.callbacks import History


class LunarLanderAgent:
    def __init__(self, observation_space, action_space, alpha, gamma, batch_size, deck_size,epsilon, epsilon_min, epsilon_decay):
        self.observation_space = observation_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_deck = deque(maxlen=deck_size)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.__agent_model()
        self.t_model = self.__agent_model()
        self.loss = []
        self.ep_state, self.ep_rewards, self.ep_action, self.ep_state_prime, self.ep_done = [], [], [], [], []
    def __agent_model(self):
        agent_model = Sequential()
        agent_model.add(Dense(128, input_dim=self.observation_space, activation='relu'))
        agent_model.add(Dense(128, activation='relu'))
        agent_model.add(Dense(self.action_space, activation='linear'))
        agent_model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return agent_model
    def store_memory(self, s,a,r,s_prime,done):
        self.memory_deck.append((s,a,r,s_prime,done))
    def clear_ep_vals(self):
        self.ep_state, self.ep_rewards, self.ep_action, self.ep_state_prime, self.ep_done = [], [], [], [], []
    def store_ep_vals(self, s,a,r,s_prime,done):
        self.ep_state.append(s)
        self.ep_action.append(a)
        self.ep_rewards.append(r)
        self.ep_state_prime.append(s_prime)
        self.ep_done.append(done)        
    def choose_action(self,state):
        return ( np.random.choice(self.action_space) if np.random.rand() <= self.epsilon else np.argmax(self.model.predict(state)[0]) )
    def learn(self):
        sample_idx = np.array(random.sample(self.memory_deck, self.batch_size))
        _s, _a, _r, _s_prime = sample_idx[:, 0],sample_idx[:, 1],sample_idx[:, 2],sample_idx[:, 3]
        _r[np.nonzero(sample_idx[:, 4] == False)] += np.multiply(self.gamma, \
                                  self.t_model.predict(np.vstack(_s_prime))[np.nonzero(sample_idx[:, 4] == False), np.argmax(self.model.predict(np.vstack(_s_prime))[np.nonzero(sample_idx[:, 4] == False), :][0],axis=1)][0])
        _target = self.model.predict(np.vstack(_s))
        _target[range(self.batch_size),np.array(_a, dtype=int)] = _r
        history = self.model.fit(np.vstack(_s), _target,verbose=0)
        return history

def train(params,view_model=False):
    env = gym.make('LunarLander-v2')
    observation_space, action_space = env.observation_space.shape[0], env.action_space.n
    agent = LunarLanderAgent(observation_space, action_space,params["alpha"], params["gamma"], \
                             params["batch_size"], params["deck_size"],params["epsilon"], \
                             params["epsilon_min"], params["epsilon_decay"])
    if view_model:
        env = wrappers.Monitor(env, "lunarlander3", force=True)
        agent.model = load_model(params["model_name"])
        episodes = 100
        agent.epsilon = 0
    else:
        env = wrappers.Monitor(env, "lunarlander3", force=True, video_callable=False)    
    mean_reward = deque(maxlen=100)
    ep_error = []
    ret_episodes, ret_reward, ret_mean_reward, ret_epsilon = [],[],[],[]
    for i in range(params["episodes"]):
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        done = False
        t = 0
        agent.loss = []
        agent.clear_ep_vals()
        while not done:
            t += 1
            if view_model: 
                env.render()
            action = agent.choose_action(state) 
            state_prime, reward, done, info = env.step(action)
            agent.store_ep_vals(state,action,state_prime, reward, done)
            state_prime = np.reshape(state_prime, [1, observation_space])
            if not view_model:
                agent.store_memory(state,action,reward,state_prime,done)
                if (len(agent.memory_deck) > agent.batch_size) : 
                    history = agent.learn()
                    agent.loss.append(history.history['loss'])
                    ep_error.append(np.mean(np.mean(agent.loss)))
            state = state_prime
            if done:
                if not view_model: agent.t_model.set_weights(agent.model.get_weights())
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        mean_reward.append(np.sum(agent.ep_rewards))
        ret_episodes.append(i)
        ret_reward.append(np.sum(agent.ep_rewards))
        ret_mean_reward.append(np.mean(mean_reward))
        ret_epsilon.append(agent.epsilon)
        if (i % 100 == 0):
            print('episode: ', i, ' epsilon: ', '%.2f' % agent.epsilon, ' score: ', '%.2f' % np.sum(agent.ep_rewards), ' mean_reward: ', '%.2f' % np.mean(mean_reward))
    if not view_model:
        agent.model.save(params["model_name"])
    return ret_episodes, ret_reward, ret_mean_reward, ret_epsilon


def rolling_mean(ary, n) :
    ret = np.cumsum(ary, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def analyze_hyperparameters():
    # Alpha Parameters Tunning
    dict_hyper = {}
    alpha = []
    hyper_parameters = {"alpha" : 0.0001, "gamma" : 0.99, "batch_size" : 30, "deck_size" : 3000 , \
                        "epsilon" : 1, "epsilon_min" : 0.01 , "epsilon_decay" : 0.995, "episodes" : 1000,  \
                        "model_name" : "lunar-lander_model3-Analyze-Alpha.h5"  }

    hyper_parameters["alpha"] = 0.0001
    episodes, e_rewards, mean_reward, epsilon = train(hyper_parameters,view_model=False)
    #print(episodes, e_rewards, mean_reward, epsilon)
    alpha.append(hyper_parameters["alpha"])
    dict_hyper[hyper_parameters["alpha"]] = e_rewards

    hyper_parameters["alpha"] = 0.001
    episodes, e_rewards, mean_reward, epsilon = train(hyper_parameters,view_model=False)
    #print(episodes, e_rewards, mean_reward, epsilon)
    alpha.append(hyper_parameters["alpha"])
    dict_hyper[hyper_parameters["alpha"]] = e_rewards
    
    hyper_parameters["alpha"] = 0.01
    episodes, e_rewards, mean_reward, epsilon = train(hyper_parameters,view_model=False)
    #print(episodes, e_rewards, mean_reward, epsilon)
    alpha.append(hyper_parameters["alpha"])
    dict_hyper[hyper_parameters["alpha"]] = e_rewards
    
    hyper_parameters["alpha"] = 0.003
    episodes, e_rewards, mean_reward, epsilon = train(hyper_parameters,view_model=False)
    #print(episodes, e_rewards, mean_reward, epsilon)
    alpha.append(hyper_parameters["alpha"])
    dict_hyper[hyper_parameters["alpha"]] = e_rewards
    
    hyper_parameters["alpha"] = 0.0005
    episodes, e_rewards, mean_reward, epsilon = train(hyper_parameters,view_model=False)
    #print(episodes, e_rewards, mean_reward, epsilon)
    alpha.append(hyper_parameters["alpha"])
    dict_hyper[hyper_parameters["alpha"]] = e_rewards

    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(211)
    ax1.get_xaxis().set_visible(False)     
    for _alpha in alpha:
        plt.plot(episodes,dict_hyper[_alpha], '',label='α = {}'.format(_alpha))
    #plt.xlim([0,len(episodes)])
    #plt.xticks(np.linspace(0,len(episodes),len(episodes)/5))
    plt.ylabel('Rewards',size=15)
    plt.xlabel('Episodes',size=15)
    plt.title('Hyperparameters - Rewards vs Episodes for α (λ = 0.99)',size=15)
    plt.legend()
    
    ax2 = fig.add_subplot(212, sharex=ax1)
    for _alpha in alpha:
        plt.plot(range(49,len(episodes)),rolling_mean(dict_hyper[_alpha],50), '',label='Rolling Mean for α = {}'.format(_alpha))
    plt.ylabel('Rewards',size=15)
    plt.xlabel('Episodes',size=15)
    plt.legend()
    fig.subplots_adjust(hspace=0)
    
    plt.show()
    plt.savefig("Alpha-Hpyerparameters3.png")
    plt.clf()

    # Gamma Parameters Tunning
    dict_hyper = {}
    gamma = []
    hyper_parameters = {"alpha" : 0.0001, "gamma" : 0.99, "batch_size" : 30, "deck_size" : 3000 , \
                        "epsilon" : 1, "epsilon_min" : 0.01 , "epsilon_decay" : 0.995, "episodes" : 1000,  \
                        "model_name" : "lunar-lander_model3-Analyze-Gamma.h5"  }
    
    hyper_parameters["gamma"] = 0.99
    episodes, e_rewards, mean_reward, epsilon = train(hyper_parameters,view_model=False)
    #print(episodes, e_rewards, mean_reward, epsilon)
    gamma.append(hyper_parameters["gamma"])
    dict_hyper[hyper_parameters["gamma"]] = e_rewards
    
    hyper_parameters["gamma"] = 0.95
    episodes, e_rewards, mean_reward, epsilon = train(hyper_parameters,view_model=False)
    #print(episodes, e_rewards, mean_reward, epsilon)
    gamma.append(hyper_parameters["gamma"])
    dict_hyper[hyper_parameters["gamma"]] = e_rewards
    
    hyper_parameters["gamma"] = 0.80
    episodes, e_rewards, mean_reward, epsilon = train(hyper_parameters,view_model=False)
    #print(episodes, e_rewards, mean_reward, epsilon)
    gamma.append(hyper_parameters["gamma"])
    dict_hyper[hyper_parameters["gamma"]] = e_rewards
    
    hyper_parameters["gamma"] = 0.90
    episodes, e_rewards, mean_reward, epsilon = train(hyper_parameters,view_model=False)
    #print(episodes, e_rewards, mean_reward, epsilon)
    gamma.append(hyper_parameters["gamma"])
    dict_hyper[hyper_parameters["gamma"]] = e_rewards
    
    hyper_parameters["gamma"] = 0.93
    episodes, e_rewards, mean_reward, epsilon = train(hyper_parameters,view_model=False)
    #print(episodes, e_rewards, mean_reward, epsilon)
    gamma.append(hyper_parameters["gamma"])
    dict_hyper[hyper_parameters["gamma"]] = e_rewards

    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(211)
    ax1.get_xaxis().set_visible(False)     
    for _gamma in gamma:
        plt.plot(episodes,dict_hyper[_gamma], '',label='λ = {}'.format(_gamma))
    #plt.xlim([0,len(episodes)])
    #plt.xticks(np.linspace(0,len(episodes),5))
    plt.ylabel('Rewards',size=15)
    plt.xlabel('Episodes',size=15)
    plt.title('Hyperparameters - Rewards vs Episodes for λ (α = 0.0001)',size=15)
    plt.legend()
    
    ax2 = fig.add_subplot(212, sharex=ax1)
    for _alpha in alpha:
        plt.plot(range(49,len(episodes)),rolling_mean(dict_hyper[_gamma],50), '',label='Rolling Mean for λ = {}'.format(_gamma))
    plt.ylabel('Rewards',size=15)
    plt.xlabel('Episodes',size=15)
    plt.legend()
    fig.subplots_adjust(hspace=0)    
    plt.show()
    plt.savefig("Gamma-Hpyerparameters3.png")
    plt.clf()
    
    
def train_model():
    dict_hyper = {}
    alpha = []
    hyper_parameters = {"alpha" : 0.0001,"gamma" : 0.99, "batch_size" : 30, "deck_size" : 3000 , \
                         "epsilon" : 1, "epsilon_min" : 0.01 , "epsilon_decay" : 0.995, "episodes" : 2000,  \
                         "model_name" : "lunar-lander_model3-Train.h5"  } 
    episodes, e_rewards, mean_reward, epsilon = train(hyper_parameters,view_model=False)
    #print(episodes, e_rewards, mean_reward, epsilon)
    alpha.append(hyper_parameters["alpha"])
    dict_hyper[hyper_parameters["alpha"]] = e_rewards
    
    fig = plt.figure(figsize=(10,6)) 
    ax1 = fig.add_subplot(211)
    ax1.get_xaxis().set_visible(False)    
    for _alpha in alpha:
        plt.plot(episodes,dict_hyper[_alpha], '',label='α = {}'.format(_alpha))
    plt.plot(episodes,mean_reward, '',label='Rolling Mean')
    #plt.xlim([0,len(episodes)])
    #plt.xticks(np.linspace(0,len(episodes),len(episodes)/5))
    plt.ylabel('Rewards',size=15)
    plt.xlabel('Episodes',size=15)
    plt.title('Training Rate for α (λ = 0.99)',size=15)
    plt.legend()
    
    ax2 = fig.add_subplot(212, sharex=ax1)
    plt.plot(episodes,epsilon, '',label='epsilon')
    plt.ylabel('Epsilon',size=15)
    plt.xlabel('Episodes',size=15)
    plt.legend()
    fig.subplots_adjust(hspace=0)
    plt.show()
    plt.savefig("Train_Mode3.png")
    plt.clf()
    
def view_model():
    dict_hyper = {}
    alpha = []
    hyper_parameters = {"alpha" : 0.0001,"gamma" : 0.99, "batch_size" : 30, "deck_size" : 10 , \
                         "epsilon" : 1, "epsilon_min" : 0.01 , "epsilon_decay" : 0.995, "episodes" : 100,  \
                         "model_name" : "lunar-lander_model3-Train.h5"  } 
    episodes, e_rewards, mean_reward, epsilon = train(hyper_parameters,view_model=True)
    #print(episodes, e_rewards, mean_reward, epsilon)
    alpha.append(hyper_parameters["alpha"])
    dict_hyper[hyper_parameters["alpha"]] = mean_reward
    
    plt.figure(figsize=(10,6))    
    for _alpha in alpha:
        plt.plot(episodes,dict_hyper[_alpha], '',label='α = {}'.format(_alpha))
    plt.plot(range(9,len(episodes)),rolling_mean(dict_hyper[0.0001],10), '',label='Rolling Mean')
    #plt.xlim([0,len(episodes)])
    #plt.xticks(np.linspace(0,len(episodes),len(episodes)/5))
    plt.ylabel('Rewards',size=15)
    plt.xlabel('Episodes',size=15)
    plt.title('Trained Agent Reward (λ = 0.99)',size=15)
    plt.legend()
    plt.show()
    plt.savefig("Train_Performance3.png")
    plt.clf()
    
if __name__ == "__main__":
    train_model()
    view_model()
    #analyze_hyperparameters()
