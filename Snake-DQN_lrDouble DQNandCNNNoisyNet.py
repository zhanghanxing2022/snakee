import argparse
import matplotlib.pyplot as plt
import sys
import time
from replay_buffer import ReplayMemory
from collections import deque
from Game import GameEnvironment
from model import QNetwork, get_network_input2, NoisyNet, QCNNNoisyNet
import os
import random
import numpy as np
import torch
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description='Snake Game DQN Training')
    # 已有的参数
    parser.add_argument('--gridsize', type=int, default=15)
    parser.add_argument('--num_episodes', type=int, default=1200)
    parser.add_argument('--target_update_frequency', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_updates', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_games', type=int, default=30)
    parser.add_argument('--checkpoint_dir', type=str,
                        default='./noisynet_1.4_Tr1e-3_tgt5_iter20')

    # 新增的参数
    parser.add_argument('--noise', type=float, default=0.1,
                        help='Noise level for NoisyNet')
    parser.add_argument('--num_bits', type=int, default=8,
                        help='Number of bits for quantization')
    parser.add_argument('--num_bits_weight', type=int, default=8,
                        help='Number of bits for weight quantization')
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['step', 'cosine', 'plateau'],
                        help='Learning rate scheduler type')
    parser.add_argument('--lr_step_size', type=int, default=500,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='Gamma for StepLR scheduler')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='Minimum learning rate for CosineAnnealingLR')
    return parser.parse_args()


args = parse_args()


class Args:
    def __init__(self, args):
        # 网络结构参数
        self.input_dim = 10
        self.hidden_dim = 20
        self.output_dim = 5
        self.metric_type = 'power'
        self.use_moving_avg = True
        self.metric_threshold = 0.5
        self.beta = 0.99

        # 从命令行参数获取
        self.noise = args.noise
        self.num_bits = args.num_bits
        self.num_bits_weight = args.num_bits_weight

        # 死亡惩罚相关参数
        self.min_death_penalty = -1
        self.max_death_penalty = -10
        self.death_penalty_steps = args.num_episodes


def get_death_penalty(episode, args):
    """计算动态死亡惩罚值"""
    x = episode / args.death_penalty_steps * 10  # 将episode映射到[0,10]范围
    # sigmoid函数实现S形变化
    sigmoid = 1 / (1 + np.exp(-x + 5))  # 中点在x=5处
    # 将sigmoid值映射到[min_penalty, max_penalty]范围
    penalty = args.min_death_penalty + \
        (args.max_death_penalty - args.min_death_penalty) * sigmoid
    return penalty


# 设备选择逻辑
# 创建目录
dir = args.checkpoint_dir
if not os.path.exists(dir):
    os.makedirs(dir)

# 初始化日志
log = open(os.path.join(dir, "log.txt"), "w+", buffering=1)
sys.stdout = log
sys.stderr = log


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('cpu')
    return torch.device('cpu')


if not os.path.exists(dir):
    os.mkdir(dir)
epsilon = 0.1
gridsize = 15
GAMMA = 0.9
device = get_device()
print(f"Using device: {device}")

# 使用 args 初始化参数
target_update_frequency = args.target_update_frequency
num_episodes = args.num_episodes
num_updates = args.num_updates
batch_size = args.batch_size
games_in_episode = args.num_games
gridsize = args.gridsize
GAMMA = 0.9

# 定义目标网络并复制参数
model = QCNNNoisyNet(input_channels=4, gridsize=gridsize,
                     output_dim=5).to(device)
target_model = QCNNNoisyNet(
    input_channels=4, gridsize=gridsize, output_dim=5).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()  # 将目标网络设为评估模式以避免梯度更新
board = GameEnvironment(gridsize, nothing=0, dead=-1, apple=1)
memory = ReplayMemory(1000)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


# 根据参数选择学习率调度器
if args.lr_scheduler == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=args.lr_step_size, 
        gamma=args.lr_gamma
    )
elif args.lr_scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_episodes,
        eta_min=args.lr_min
    )
elif args.lr_scheduler == 'plateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=args.lr_gamma,
        patience=100,
        min_lr=args.lr_min
    )
def run_episode(num_games, current_episode):
    run = True
    move = 0
    games_played = 0
    total_reward = 0
    episode_games = 0
    len_array = []
    current_death_penalty = get_death_penalty(current_episode, Args(args))
    board.reward_dead = current_death_penalty
    while run:
        state = get_network_input2(board.snake, board.apple)
        with torch.no_grad():
            action_values = model(state)
            action = torch.argmax(action_values).item()

        reward, done, len_of_snake = board.update_boardstate(action)
        next_state = get_network_input2(board.snake, board.apple)

        memory.push(state.cpu(), action, reward, next_state.cpu(), done)
        total_reward += reward
        episode_games += 1
        if board.game_over:
            games_played += 1
            len_array.append(len_of_snake)
            board.resetgame()
            if num_games == games_played:
                run = False

    avg_len_of_snake = np.mean(len_array)
    max_len_of_snake = np.max(len_array)
    return total_reward, avg_len_of_snake, max_len_of_snake


# 设置目标网络更新频率

MSE = nn.MSELoss()


def learn(num_updates, batch_size):
    # 确保buffer中有足够的样本
    if len(memory) < batch_size:
        return 0.0  # 如果样本不足，跳过学习
    total_loss = 0
    for i in range(num_updates):
        model.reset_noise()
        target_model.reset_noise()

        optimizer.zero_grad()
        sample = memory.sample(batch_size)
        states, actions, rewards, next_states, dones = sample  # 将数据移到正确的设备上
        states = torch.stack(states).to(device)
        next_states = torch.stack(next_states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)

        q_local = model(states)
        next_q_value = target_model(next_states)

        Q_expected = q_local.gather(1, actions.unsqueeze(1)).squeeze(1)
        Q_targets_next = torch.max(next_q_value, 1)[0] * (1 - dones)
        Q_targets = rewards + GAMMA * Q_targets_next

        loss = MSE(Q_expected, Q_targets)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        if i % target_update_frequency == 0:
            target_model.load_state_dict(model.state_dict())

    return total_loss


print_every = 10


def train():

    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []

    avg_len_array = []
    avg_max_len_array = []

    time_start = time.time()

    for i_episode in range(num_episodes+1):

        # print('i_episode: ', i_episode)

        score, avg_len, max_len = run_episode(games_in_episode, i_episode)

        scores_deque.append(score)
        scores_array.append(score)
        avg_len_array.append(avg_len)
        avg_max_len_array.append(max_len)

        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)

        total_loss = learn(num_updates, batch_size)


        dt = (int)(time.time() - time_start)

        if i_episode % print_every == 0 and i_episode > 0:
            print('Ep.: {:6}, Loss: {:.3f}, Avg.Score: {:.2f}, Avg.LenOfSnake: {:.2f}, '
                  'Max.LenOfSnake: {:.2f}, Death Penalty: {:.2f}, LR: {:.2e}, Time: {:02}:{:02}:{:02}'.format(
                      i_episode, total_loss, score, avg_len, max_len, board.reward_dead, optimizer.param_groups[0]['lr'],
                      dt//3600, dt % 3600//60, dt % 60))

        memory.truncate()
        # 更新学习率
        if args.lr_scheduler == 'plateau':
            scheduler.step(avg_score)  # 使用平均分数作为指标
        else:
            scheduler.step()
        if i_episode % 250 == 0 and i_episode > 0:
            torch.save(model.state_dict(), os.path.join(
                dir, f"Snake_{i_episode}"))

    return scores_array, avg_scores_array, avg_len_array, avg_max_len_array


scores, avg_scores, avg_len_of_snake, max_len_of_snake = train()

print('length of scores: ', len(scores),
      ', len of avg_scores: ', len(avg_scores))
