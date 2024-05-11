#!pip install gym
!pip install setuptools==65.5.0 pip==21
!pip install stable_baselines3
!pip install gym_anytrading
!pip install finta
!pip install 'shimmy>=0.2.1'



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np
import random

class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # input layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # hidden layer
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # output layer

    def forward(self, x):
        # activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # capacity of buffer
        self.buffer = []  # replay buffer
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        ''' replay buffer is a queue (LIFO)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DQN:
    def __init__(self, state_dim, action_dim, cfg):

        self.action_dim = action_dim
        self.device = cfg.device  # cpu or gpu
        self.gamma = cfg.gamma  # discount factor
        self.frame_idx = 0  # attenuation
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
                                         (cfg.epsilon_start - cfg.epsilon_end) * \
                                         math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.batch_size = cfg.batch_size
        self.policy_net = MLP(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = MLP(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        for target_param, param in zip(self.target_net.parameters(),
                                       self.policy_net.parameters()):  # copy parameters to target net
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr) # optimizer
        self.memory = ReplayBuffer(cfg.memory_capacity)  # experience replay

    def choose_action(self, state):
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                state = torch.tensor([state], device=self.device, dtype=torch.float32)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item()  # choose the action with maximum q-value
        else:
            action = random.randrange(self.action_dim)
        return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        # transfer to tensor
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        # calculate the expected q-value, for final state, done_batch[0]=1 and the corresponding
        # expected_q_value equals to reward
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))
        # update the network
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():  # avoid gradient explosion by using clip
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save(self, path):
        torch.save(self.target_net.state_dict(), path + 'dqn_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path + 'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)



class TradingSystem_v0:
    def __init__(self, returns_data, k_value, mode):
        self.mode = mode  # test or train
        self.index = 0
        self.data = returns_data
        self.tickers = list(returns_data.keys())
        self.current_stock = self.tickers[self.index]
        self.r_ts = self.data[self.current_stock]
        self.k = k_value
        self.total_steps = len(self.r_ts) - self.k
        self.current_step = 0
        self.initial_state = tuple(self.r_ts[:self.k])  # Use tuple because it's immutable
        self.state = self.initial_state
        self.reward = 0.0
        self.is_terminal = False

    # write step function that returns obs(next state), reward, is_done
    def step(self, action):
        self.current_step += 1
        if self.current_step == self.total_steps:
            self.is_terminal = True
        self.reward = (action-1) * self.r_ts[self.current_step + self.k - 1]
        self.state = tuple(self.r_ts[self.current_step:(self.k + self.current_step)])
        return self.state, self.reward, self.is_terminal

    def reset(self):
        if self.mode == 'train':
            self.current_stock = random.choice(self.tickers)  # randomly pick a stock for every episode
        else:
            self.current_stock = self.tickers[self.index]
            self.index += 1
        self.r_ts = self.data[self.current_stock]
        self.total_steps = len(self.r_ts) - self.k
        self.current_step = 0
        self.initial_state = tuple(self.r_ts[:self.k])
        self.state = self.initial_state
        self.reward = 0.0
        self.is_terminal = False
        return self.state

import gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv
from finta import TA
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Stable baselines - rl stuff
import yfinance as yf
import time
# stable-baselines is a newer framework for reinforcement learning, uses pytorch (Tensors and neural networks w/ GPU acceleration)
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
import os
curr_path = os.path.dirname("ModelOutput")
import gym
import torch
import random
import datetime as dt
#from dqn import DQN
#from trader_environment import TradingSystem_v0

curr_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
random.seed(11)

# Define the tickers and the time range
start_date = dt.date(2020, 1, 1)
#start_date = dt.date(2023, 1, 1)
end_date = dt.datetime.today().strftime ('%Y-%m-%d')
train_tickers = ['INTC','META','NFLX','GOOG','PINS','AMD','MSTR','WFC','GS']
#train_tickers = ['AMD','NVDA','TSLA','AAPL']
test_tickers = ['AVGO','NVDA','CSCO','AAPL']


class Config:
    '''
    hyperparameters
    '''

    def __init__(self):
        ################################## env hyperparameters ###################################
        self.algo_name = 'DQN' # algorithmic name
        self.env_name = 'TradingSystem_v0' # environment name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # examine GPU
        self.seed = 11 # random seed
        self.train_eps = 100 # training episodes
        self.state_space_dim = 50 # state space size (K-value)
        self.action_space_dim = 3 # action space size (short: 0, neutral: 1, long: 2)
        ################################################################################

        ################################## algo hyperparameters ###################################
        self.gamma = 0.95  # discount factor
        self.epsilon_start = 0.90  # start epsilon of e-greedy policy
        self.epsilon_end = 0.01  # end epsilon of e-greedy policy
        self.epsilon_decay = 500  # attenuation rate of epsilon in e-greedy policy
        self.lr = 0.0001  # learning rate
        self.memory_capacity = 1000  # capacity of experience replay
        self.batch_size = 64  # size of mini-batch SGD
        self.target_update = 4  # update frequency of target network
        self.hidden_dim = 128  # dimension of hidden layer
        ################################################################################

        ################################# save path ##############################
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'
        self.save = False #True  # whether to save the image
        ################################################################################


def env_agent_config(data, cfg, mode):
    ''' create environment and agent
    '''
    env = TradingSystem_v0(data, cfg.state_space_dim, mode)
    agent = DQN(cfg.state_space_dim, cfg.action_space_dim, cfg)
    if cfg.seed != 0:  # set random seeds
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
    return env, agent


def train(cfg, env, agent):
    ''' training
    '''
    print('Start Training!')
    print(f'Environment：{cfg.env_name}, Algorithm：{cfg.algo_name}, Device：{cfg.device}')
    rewards = []  # record total rewards
    ma_rewards = []  # record moving average total rewards
    for i_ep in range(cfg.train_eps):
        ep_reward = 0
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)  # save transition
            state = next_state
            agent.update()
            ep_reward += reward
            if done:
                break
        if (i_ep + 1) % cfg.target_update == 0:  # update target network
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print('Episode：{}/{}, Reward：{}'.format(i_ep + 1, cfg.train_eps, ep_reward))
    print('Finish Training!')
    return rewards, ma_rewards


def test(cfg, env, agent):
    print('Start Testing!')
    print(f'Environment：{cfg.env_name}, Algorithm：{cfg.algo_name}, Device：{cfg.device}')
    ############# Test does not use e-greedy policy, so we set epsilon to 0 ###############
    cfg.epsilon_start = 0.0
    cfg.epsilon_end = 0.0
    ################################################################################
    stocks = env.tickers
    rewards = []  # record total rewards
    for i_ep in range(len(stocks)):
        ep_reward = 0
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        print(f"Episode：{i_ep + 1}/{len(stocks)}，Reward：{ep_reward:.1f}")
    print('Finish Testing!')
    return stocks, rewards


if __name__ == "__main__":

    # download stock data from yahoo finance
    train_data = {}
    for ticker in train_tickers:
        data = yf.download(ticker, start_date, end_date, interval='1d')
        returns = data['Adj Close'].pct_change()[1:]
        train_data[ticker] = returns

    test_data = {}
    for ticker in test_tickers:
        data = yf.download(ticker, start_date, end_date, interval='1d')
        returns = data['Adj Close'].pct_change()[1:]
        test_data[ticker] = returns


    cfg = Config()
    # training
    env, agent = env_agent_config(train_data, cfg, 'train')
    rewards, ma_rewards = train(cfg, env, agent)
    os.makedirs(cfg.result_path)  # create output folders
    os.makedirs(cfg.model_path)
    agent.save(path=cfg.model_path)  # save model
    #fig, ax = plt.subplots(1, 1, figsize=(10, 7))   # plot the training result
    #ax.plot(list(range(1, cfg.train_eps+1)), rewards, color='blue', label='rewards')
    #ax.plot(list(range(1, cfg.train_eps+1)), ma_rewards, color='green', label='ma_rewards')
    #ax.legend()
    #ax.set_xlabel('Episode')
    #plt.savefig(cfg.result_path+'train.jpg')
    #plt.show()

    # testing
    all_data = {**train_data, **test_data}
    env, agent = env_agent_config(all_data, cfg, 'test')
    agent.load(path=cfg.model_path)  # load model
    stocks, rewards = test(cfg, env, agent)
    buy_and_hold_rewards = [sum(all_data[stock]) for stock in stocks]
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))  # plot the test result
    width = 0.3
    x = np.arange(len(stocks))
    ax.bar(x, rewards, width=width, color='salmon', label='DQN')
    ax.bar(x+width, buy_and_hold_rewards, width=width, color='orchid', label='Buy and Hold')
    ax.set_xticks(x+width/2)
    ax.set_xticklabels(stocks, fontsize=12)
    ax.legend()
    plt.savefig(cfg.result_path+'test.jpg')
    plt.show()


########################################################################################################################

for ticker in test_tickers:

    #df = yf.download(ticker, period="90d", interval="1d")
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    del df['Adj Close']
    df = df.reset_index()
    print(df)

    df['Date'] = pd.to_datetime(df['Date'])
    print(df.dtypes)

    df.sort_values('Date', ascending=True, inplace=True)
    print(df.head())

    df.set_index('Date', inplace=True)
    print(df.head())

    env = gym.make('stocks-v0', df=df, frame_bound=(5, 250), window_size=5)
    print(env.signal_features)
    print(env.action_space)

    # BUILD ENVIRONMENT:
    state = env.reset()
    while True:
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        if done:
            print("info", info)
            break
    # plt.figure(figsize=(15, 6))
    # plt.cla()
    # env.render_all()
    # plt.show()
    # time.sleep(3000)
    # Adding technicals:
    # df['Volume'] = df['Volume'].apply(lambda x: float(x.replace(",", "")))
    df['SMA'] = TA.SMA(df, 12)
    df['RSI'] = TA.RSI(df)
    df['OBV'] = TA.OBV(df)
    df.fillna(0, inplace=True)


    # print(df)
    # Create a new environment:
    def add_signals(env):
        start = env.frame_bound[0] - env.window_size
        end = env.frame_bound[1]
        prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
        signal_features = env.df.loc[:, ['Low', 'Volume', 'SMA', 'RSI', 'OBV']].to_numpy()[start:end]
        return prices, signal_features


    class MyCustomEnv(StocksEnv):
        _process_data = add_signals


    env2 = MyCustomEnv(df=df, window_size=12, frame_bound=(12, 50))

    # Build environment and train:
    env_maker = lambda: env2
    env = DummyVecEnv([env_maker])
    model = A2C('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)  # total_timesteps=1000000

    # Evaluate the model:
    env = MyCustomEnv(df=df, window_size=12, frame_bound=(80, 250))
    obs = env.reset()
    while True:
        obs = obs[np.newaxis, ...]
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            print("Reward & Total RL AI Strategy Return for:", ticker)
            #print("OUTPUT", info)
            print("Total Reward for",ticker)
            print(info['total_reward'])
            print("Total Backtest Return for:",ticker)
            bt_return = info['total_profit']
            bt_return_final = bt_return - int(bt_return)
            print(bt_return_final)
            break

    time.sleep(10)
    #plt.figure(figsize=(15, 6))
    #plt.cla()
    #env.render_all()
    #plt.show()