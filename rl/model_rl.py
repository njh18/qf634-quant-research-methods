import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pylab as plt
import random
from collections import namedtuple
from keras.layers import Input, Dense, Flatten, Dropout, Conv1D, MaxPooling1D
from keras.models import Sequential, Model
from sklearn.preprocessing import StandardScaler
import pickle as pkl
import itertools

class RLEnvironment():
    def __init__(self
                 , input_file='data/df_price_train.csv'
                 , observe_window = 90
                 , rebalance_window = 30):  
        self.input_file = input_file
        self.prices = None # numpy array
        self.returns = None # numpy array
        self.asset_names = None # numpy array
        self.observe_window = observe_window
        self.rebalance_window = rebalance_window
        self.t = None # current time
        self.T = None
        self.n_asset = None
        self.done = False

        self._load_data() 
        self.reset()

    def _load_data(self):
        data = pd.read_csv(self.input_file)
        data = data.drop(columns = ['Date'])
        # data2 = data.pct_change().dropna()
        # self.prices = data.values
        # self.returns = data.values
        self.asset_names = data.columns
        self.T = data.shape[0]
        self.n_asset = data.shape[1]
        self.scaler = StandardScaler().fit(data.values)
        self.prices = self.scaler.transform(data.values)[1:]
        self.returns = data.pct_change().dropna().values
    
    def _reset(self, t=None):
        if t is not None:
            self.t = t
        else: 
            self.t = self.observe_window
        self.done = False
        state = self.check_state()
        return state
    
    def reset(self, t=None):
        return self._reset(t)
    
    def random_reset(self):
        self.t = random.randint(self.observe_window
                                , self.T - self.rebalance_window - 1)
        return self._reset(self.t)

    def check_state(self):
        assert self.observe_window <= self.t
        Pr_i_ts = self.prices[self.t - self.observe_window : min(self.t, self.T)]
        return Pr_i_ts
    
    def check_reward(self, action):
        W = self.normalize(action)
        R_i_ts = self.returns[self.t - self.rebalance_window : min(self.t, self.T)] # daily returns of the individual assets
        R_i = np.mean(R_i_ts, axis=0)  # mean returns of the individual assets
        # V = np.cov(R_i_ts.T)  
        R_p_ts = np.dot(R_i_ts, W)  # daily returns of the portfolio
        r_p = np.sum(np.dot(R_i, W))
        # v_p = np.dot(W.T, np.dot(V, W))
        # sharpe_ratio = r_p / np.sqrt(v_p)  # Sharpe ratio of the portfolio
        # semi_v_p = np.mean(np.minimum(R_p_ts, 0)**2)+1e-4
        # sortino_ratio = r_p / np.sqrt(semi_v_p)  # Sortino ratio of the portfolio
        # cum_r_p = np.prod(R_p_ts+1)-1
        return R_p_ts, r_p # sharpe_ratio
    
    def step(self, action):
        reward = ([], None)
        next_state = []
        if (self.t + self.rebalance_window) >= self.T - 1:
            self.done = True          
        self.t += self.rebalance_window
        next_state = self.check_state()
        reward = self.check_reward(action)  # reward from next state
        return (reward, self.done, next_state)
    
    def normalize(self, action):
        w_min = np.min(action)
        w_sum = np.sum(action - w_min)
        if np.isclose(1.0, w_sum):
            return action - w_min
        elif w_sum == 0:
            return np.ones(self.n_asset) / self.n_asset
        else:
            np.seterr(all='raise')  
            # This will raise exceptions for all warnings
            try:
                return (action - w_min) / w_sum
            except Exception as e:
                print(f"{e}\nw_sum={w_sum}\n{action}")

class QEstimator():
    """Q-Value Estimator neural network.

    """
    def __init__(self, n_asset, obs_window, n_action):
        
        self.n_asset = n_asset
        self.obs_window = obs_window
        self.n_action = n_action
        self.model = []
        
        for i in range(self.n_asset):
            self.model.append(self._build())

    def _build(self):
        
        model = Sequential()
        model.add(Conv1D(filters=64,kernel_size=8,activation='relu',input_shape=(self.obs_window,1),padding='valid'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=64,kernel_size=8,activation='relu',input_shape=(self.obs_window,1),padding='valid'))
        model.add(Flatten())
        model.add(Dense(150,activation='relu'))
        model.add(Dense(self.n_action, activation='linear'))
        model.compile(optimizer='adam',loss='mse')
        return model
    
    def predict(self, state):
        q_values = []
        for i in range(self.n_asset):
            q_values.append(self.model[i].predict(np.expand_dims(state[:,i], 0), verbose=0))
        
        return q_values

    def update(self, state, Q):
        for i in range(self.n_asset):
            self.model[i].fit(np.expand_dims(state[:,i], 0)
                        , Q[i]
                        , epochs=1, verbose=0)

class RLAgent():

    def __init__(self, n_action):
        self.n_action = n_action
        self.n = int((self.n_action-1)/2) # discrete weight into n --> min weight is 1/n
        # self.env = env
        # self.n_asset = env.n_asset
        # self.observe_window = env.observe_window
        
        # Q estimator: NN model that generates Q. Q is a map of state -> action-values. Each value is a numpy array of length nA.
        self.q_estimator = None

    def uni_policy(self):
        """
        Creates a policy function w/ output action of same asset weights.
        
        Args:
            nA: Number of actions in the environment.
        
        Returns:
            A function that takes an observation as input and returns a vector of action probabilities
        """
        A = np.ones(self.n_action, dtype=float) / self.n_action
        def policy_func(state):
            return A
        return policy_func

    def epsilon_greedy_policy(self):
        """
        Creates an epsilon-greedy policy based on a given Q-generating model.
        
        Args:
            n_asset: Number of assets in the portfolio
        
        Returns:
            A function that takes q_estimator, state and epsilon as arguments and returns actions
                q_estimator: NN model that generates Q. Q is a map of state -> action-values. Each value is a numpy array of length nA.
                epsilon: The probability to select a random action. Float between 0 and 1.
        
        """
        def generate_actions(Q): 

            Q = np.squeeze(Q)  # (n_assets, 1, n_actions) -> (n_assets, n_actions)
            action_argmax = np.argmax(Q, axis=-1)  # (n_assets,) chosen action according to the max action value
            actions = np.where(action_argmax>self.n, self.n-action_argmax, action_argmax) # 0-hold, 1-30 buy, 31-60 sell

            return actions
        
        def policy_func(state, q_estimator: QEstimator, epsilon):
            
            Q = q_estimator.predict(state)
            best_action = generate_actions(Q)
            random_action = np.random.randint(-self.n, self.n, size=(q_estimator.n_asset,)) 
            if random.random() < epsilon:
                action = random_action
            else:
                action = best_action
            return action
        
        return policy_func
    
    def q_learn(self
                    , env: RLEnvironment
                    , n_episodes
                    , alpha=0.5
                    , epsilon=1.0
                    , epsilon_min=0.01
                    , epsilon_decay_rate=0.95
                    , gamma=0.95
                    , replay_batch_size=10
                    ):
        """
        Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy.

        Args:
            env: instance of RLEnvironment.
            num_episodes: Number of episodes to run for.
            alpha: TD learning rate.
            epsilon: Chance to sample a random action. Float between 0 and 1.
            gamma: discount factor.
            replay_batch_size: reach a certain memory size before replay. Replay is for the model to learn across a set of uncorrelated transitions https://arxiv.org/pdf/1509.02971
        
        Returns:
            stats: Episode stats.
        """

        # Keeps track of useful statistics
        EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
        stats = EpisodeStats(
            episode_lengths=np.zeros(n_episodes)
            , episode_rewards=np.zeros(n_episodes)
        )
        # Q estimator
        self.q_estimator=QEstimator(env.n_asset, env.observe_window, self.n_action) 
        # The policy we're following
        policy = self.epsilon_greedy_policy()
        # Benchmark policy (equal weights)
        policy_bm = self.uni_policy()

        # The replay memory
        replay_memory = []
        Memory = namedtuple("memory", ["state", "action", "reward", "done", "next_state"])

        for i_eps in range(n_episodes):
            returns_recorder_bm = []
            returns_recorder = []
            actions_recorder = []
            
            print("\nEpisode {}/{}, episilon={}".format(i_eps + 1, n_episodes, epsilon))
            # sys.stdout.flush()
            
            # Reset the environment
            state = env.random_reset()
            
            for t in itertools.count():
                
                # Take a step
                action = policy(state, self.q_estimator, epsilon)
                action_bm = policy_bm(state)
                reward, done, next_state = env.step(action)
                # if done:
                #     break 
                reward_bm = env.check_reward(action_bm)  #!! Note to calculate rewards for the benchmark after calling env.step() coz it is calculated based on next state.
                # Update recorders and statistics
                returns_recorder_bm.extend(reward_bm[0]) # daily returns of the benchmark portfolio
                actions_recorder.extend(action)
                returns_recorder.extend(reward[0]) # daily returns of the portfolio
                stats.episode_rewards[i_eps] += reward[1] # Sharp ratio of the portfolio
                stats.episode_lengths[i_eps] = t
                
                if len(replay_memory) >= 500:
                    replay_memory.pop(0)
                replay_memory.append(Memory(state, action, reward, done, next_state))

                if len(replay_memory) >= replay_batch_size:
                    sample_memory = random.sample(replay_memory, replay_batch_size)
                    # Temporal Difference (TD) Update
                    for memo in sample_memory:
                        # init
                        Q_reward = np.zeros((self.q_estimator.n_asset, self.q_estimator.n_action))
                        Q_next = np.zeros((env.n_asset, self.n_action))
                        # restore the chosen action which has max action value given by Q
                        action_argmax = np.where(memo.action>=0, memo.action, self.n-memo.action)
                        # give reward to the chosen action
                        np.put_along_axis(Q_reward 
                                        , action_argmax.reshape(-1,1)
                                        , memo.reward[1]  
                                        , axis=-1)
                        if not memo.done:
                            Q_next = np.squeeze(self.q_estimator.predict(memo.next_state)) 
                        best_action_next = np.max(Q_next, axis=-1)
                        Q_next = np.where(Q_next==best_action_next.reshape(-1,1), Q_next, 0)
                        Q_expect = Q_reward + gamma * Q_next
                        Q_now = np.squeeze(self.q_estimator.predict(memo.state))
                        Q_learn = (1-alpha) * Q_now + alpha * Q_expect
                        Q_learn = [A.reshape(1, -1) for A in Q_learn] # reshape to match model update function
                        # update Q function
                        self.q_estimator.update(memo.state, Q_learn)  
                
                if done:
                    break         
                # replay_memory = []
                state = next_state
                # end of one episode

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay_rate

            if i_eps % 5 == 0:
                plt.figure(figsize = (12, 2))
                plt.plot(np.cumprod(np.array(returns_recorder)+1)-1, color = 'black', ls = '-')
                plt.plot(np.cumprod(np.array(returns_recorder_bm)+1)-1, color = 'grey', ls = '--')
                plt.show()
            
            # plt.figure(figsize = (12, 2))
            # for a in actions_recorder:    
            #     plt.bar(np.arange(env.n_asset), a, color = 'grey', alpha = 0.25)
            #     plt.xticks(np.arange(env.n_asset), env.asset_names, rotation='vertical')
            # plt.show()

        return stats
    
    def save_model(self, path):
        with open(path, 'wb') as f:
            pkl.dump(self.q_estimator, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            self.q_estimator = pkl.load(f)
    
    def q_predict(self, env: RLEnvironment):
        
        Performance = namedtuple("Performance",["portfolio_returns", "portfolio_composition", "benchmark_returns"])
        performance = Performance(
            portfolio_returns=[]
            , portfolio_composition=[]
            , benchmark_returns=[]
        )

        # Reset the environment
        state = env.reset()
        # The policy we're following
        policy = self.epsilon_greedy_policy()
        # Benchmark policy (equal weights)
        policy_bm = self.uni_policy()

        for t in itertools.count():
            
            # Take a step
            action = policy(state, self.q_estimator, epsilon=0)  # set epsilon=0 to use the optimal policy
            action_bm = policy_bm(state)
            reward, done, next_state = env.step(action)
            # if done:
            #     break   
            reward_bm = env.check_reward(action_bm) 
            # Normalize the weights
            w_min = np.min(action)
            w_sum = np.sum(action - w_min)
            action = (action - w_min) / w_sum
            
            # Update recorders and statistics
            performance.benchmark_returns.extend(reward_bm[0]) # daily returns of the benchmark portfolio
            performance.portfolio_composition.append(action)
            performance.portfolio_returns.extend(reward[0]) # daily returns of the portfolio
            if done:
                break   
            state = next_state

        plt.figure(figsize = (12, 4))
        plt.plot(np.cumprod(np.array(performance.portfolio_returns)+1)-1, ls = '-')
        plt.plot(np.cumprod(np.array(performance.benchmark_returns)+1)-1, color = 'grey', ls = '--')
        plt.show()

        return performance
    
                
    
    