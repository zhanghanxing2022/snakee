import matplotlib.pyplot as plt
import sys
import time
from replay_buffer import ReplayMemory
from collections import deque
from Game import GameEnvironment
from model import QNetwork, get_network_input
import os
import random
import numpy as np
import torch
import torch.nn as nn
import argparse
print_every = 10
def parse_args():
    parser = argparse.ArgumentParser(description='Snake Game DQN Training')
    parser.add_argument('--gridsize', type=int, default=15)
    parser.add_argument('--num_episodes', type=int, default=12000)
    parser.add_argument('--target_update_frequency', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num_updates', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_games', type=int, default=30)
    parser.add_argument('--checkpoint_dir', type=str, default='./dir_chk_4')
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_min', type=float, default=0.01)
    parser.add_argument('--decay_rate', type=float, default=0.995)
    return parser.parse_args()

# 在文件开头添加
args = parse_args()
model = QNetwork(input_dim=10, hidden_dim=20, output_dim=5)

# 替换相应的硬编码值
dir = args.checkpoint_dir
if not os.path.exists(dir):
    os.makedirs(dir)

# 更新模型参数
gridsize = args.gridsize
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# 更新epsilon相关参数
epsilon_start = args.epsilon_start
epsilon_min = args.epsilon_min
decay_rate = args.decay_rate
epsilon = epsilon_start

# 更新训练参数
target_update_frequency = args.target_update_frequency
num_episodes = args.num_episodes
num_updates = args.num_updates
batch_size = args.batch_size
games_in_episode = args.num_games
if not os.path.exists(dir):
    os.mkdir(dir)
log = open(os.path.join( dir,"log.txt"), "w+", buffering=1)
sys.stdout = log
sys.stderr = log
GAMMA = 0.9

board = GameEnvironment(gridsize, nothing=0, dead=-1, apple=1)
memory = ReplayMemory(1000)






def run_episode(num_games):
    run = True
    move = 0
    games_played = 0
    total_reward = 0
    episode_games = 0
    len_array = []
    global epsilon
    epsilon = max(epsilon * decay_rate, epsilon_min)
    while run:
        state = get_network_input(board.snake, board.apple)
        action_0 = model(state)
        rand = np.random.uniform(0, 1)
        if rand > epsilon:
            action = torch.argmax(action_0)
        else:
            action = np.random.randint(0, 5)

        # update_boardstate the same snake till
        reward, done, len_of_snake = board.update_boardstate(action)
        next_state = get_network_input(board.snake, board.apple)

        memory.push(state, action, reward, next_state, done)

        total_reward += reward

        episode_games += 1

        if board.game_over == True:
            games_played += 1
            len_array.append(len_of_snake)
            board.resetgame()

            if num_games == games_played:
                run = False

    avg_len_of_snake = np.mean(len_array)
    max_len_of_snake = np.max(len_array)
    return total_reward, avg_len_of_snake, max_len_of_snake



# 定义目标网络并复制参数
target_model = QNetwork(input_dim=10, hidden_dim=20, output_dim=5)
target_model.load_state_dict(model.state_dict())
target_model.eval()  # 将目标网络设为评估模式以避免梯度更新

# 设置目标网络更新频率

MSE=nn.MSELoss()
def learn(num_updates, batch_size):
    total_loss = 0
        # 确保buffer中有足够的样本
    if len(memory) < batch_size:
        return 0.0  # 如果样本不足，跳过学习

    for i in range(num_updates):
        optimizer.zero_grad()
        sample = memory.sample(batch_size)

        states, actions, rewards, next_states, dones = sample
        states = torch.cat([x.unsqueeze(0) for x in states], dim=0)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.cat([x.unsqueeze(0) for x in next_states])
        dones = torch.FloatTensor(dones)

        q_local = model(states)
        next_q_value = target_model(next_states)  # 使用目标网络的Q值

        Q_expected = q_local.gather(1, actions.unsqueeze(
            0).transpose(0, 1)).transpose(0, 1).squeeze(0)
        Q_targets_next = torch.max(next_q_value, 1)[0] * (1 - dones)
        Q_targets = rewards + GAMMA * Q_targets_next

        loss = MSE(Q_expected, Q_targets)
        total_loss += loss
        loss.backward()
        optimizer.step()

        # 更新目标网络参数
        if i % target_update_frequency == 0:
            target_model.load_state_dict(model.state_dict())

    return total_loss



def train():

    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []

    avg_len_array = []
    avg_max_len_array = []

    time_start = time.time()

    for i_episode in range(num_episodes+1):

        # print('i_episode: ', i_episode)

        score, avg_len, max_len = run_episode(games_in_episode)

        scores_deque.append(score)
        scores_array.append(score)
        avg_len_array.append(avg_len)
        avg_max_len_array.append(max_len)

        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)

        total_loss = learn(num_updates, batch_size)

        dt = (int)(time.time() - time_start)

        if i_episode % print_every == 0 and i_episode > 0:
            print('Ep.: {:6}, Loss: {:.3f}, Avg.Score: {:.2f}, Avg.LenOfSnake: {:.2f}, Max.LenOfSnake:  {:.2f} Time: {:02}:{:02}:{:02} '.
                  format(i_episode, total_loss, score, avg_len, max_len, dt//3600, dt % 3600//60, dt % 60))

        memory.truncate()

        if i_episode % 250 == 0 and i_episode > 0:
            torch.save(model.state_dict(),os.path.join(dir,f"Snake_{i_episode}"))

    return scores_array, avg_scores_array, avg_len_array, avg_max_len_array


scores, avg_scores, avg_len_of_snake, max_len_of_snake = train()

print('length of scores: ', len(scores),
      ', len of avg_scores: ', len(avg_scores))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores, label="Score")
plt.plot(np.arange(1, len(avg_scores)+1), avg_scores,
         label="Avg score on 100 episodes")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.ylabel('Score')
plt.xlabel('Episodes #')
plt.savefig(os.path.join(dir, "scores.png"))
# plt.show()
ax1 = fig.add_subplot(121)
plt.plot(np.arange(1, len(avg_len_of_snake)+1),
         avg_len_of_snake, label="Avg Len of Snake")
plt.plot(np.arange(1, len(max_len_of_snake)+1),
         max_len_of_snake, label="Max Len of Snake")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.ylabel('Length of Snake')
plt.xlabel('Episodes #')
# plt.show()
plt.savefig(os.path.join(dir, "Length.png"))

n, bins, patches = plt.hist(
    max_len_of_snake, 45, density=1, facecolor='green', alpha=0.75)
l = plt.plot(np.arange(1, len(bins) + 1), 'r--', linewidth=1)
mu = round(np.mean(max_len_of_snake), 2)
sigma = round(np.std(max_len_of_snake), 2)
median = round(np.median(max_len_of_snake), 2)
print('mu: ', mu, ', sigma: ', sigma, ', median: ', median)
plt.xlabel('Max.Lengths, mu = {:.2f}, sigma={:.2f},  median: {:.2f}'.format(
    mu, sigma, median))
plt.ylabel('Probability')
plt.title('Histogram of Max.Lengths')
plt.axis([4, 44, 0, 0.15])
plt.grid(True)
plt.savefig(os.path.join(dir, "Max Length.png"))

# plt.show()
