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
import argparse
import multiprocessing as mp
from multiprocessing import Pool

# 全局变量定义
global_model = None
global_device = None
global_gridsize = 15

print("Starting script...")  # 调试信息
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def init_worker():
    """初始化工作进程的全局变量"""
    global global_model, global_device, global_gridsize
    global_device = get_device()
    global_model = QCNNNoisyNet(input_channels=4, gridsize=global_gridsize, output_dim=5).to(global_device)
    global_model.load_state_dict(torch.load('/Users/zhanghanxing/Desktop/work/abv/Snakeee/dir_chk_6/Snake_5000', weights_only=True))
    global_model.eval()



def parse_args():
    print("Parsing arguments...")  # 调试信息
    parser = argparse.ArgumentParser(description='Snake DQN Training')
    parser.add_argument('--gridsize', type=int, default=15)
    parser.add_argument('--num_episodes', type=int, default=1200)
    parser.add_argument('--target_update_frequency', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_updates', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_games', type=int, default=30)
    parser.add_argument('--checkpoint_dir', type=str, default='./dir_chk_4Tr1e-3_tgt5_iter20')
    parser.add_argument('--num_processes', type=int, default=5)
    return parser.parse_args()
def run_single_game(seed):
    """单个游戏进程的运行函数"""
    global global_model, global_device, global_gridsize

    # 添加错误检查
    if global_model is None or global_device is None or global_gridsize is None:
        print("Error: Global variables not properly initialized")
        return None

    try:
        local_board = GameEnvironment(global_gridsize, nothing=0, dead=-10, apple=1)
        state = get_network_input2(local_board.snake, local_board.apple)

        with torch.no_grad():
            action_values = global_model(state.to(global_device))
            action = torch.argmax(action_values).item()

        reward, done, len_of_snake = local_board.update_boardstate(action)
        next_state = get_network_input2(local_board.snake, local_board.apple)

        return state.cpu(), action, reward, next_state.cpu(), done, len_of_snake
    except Exception as e:
        print(f"Error in run_single_game: {e}")
        return None

def run_episode_parallel(memory, num_games, num_processes):
    """并行运行多个游戏实例"""
    total_reward = 0
    len_array = []

    # 创建进程池
    with Pool(processes=num_processes, initializer=init_worker) as pool:
        seeds = [random.randint(0, 10000) for _ in range(num_games)]
        results = pool.map(run_single_game, seeds)

    # 添加结果验证
    valid_results = [r for r in results if r is not None]
    if len(valid_results) == 0:
        print("Warning: No valid results returned from parallel games")
        return 0, 0, 0

    # 处理结果
    for state, action, reward, next_state, done, len_of_snake in valid_results:
        memory.push(state, action, reward, next_state, done)
        total_reward += reward
        if done:
            len_array.append(len_of_snake)

    avg_len_of_snake = np.mean(len_array) if len_array else 0
    max_len_of_snake = np.max(len_array) if len_array else 0
    return total_reward, avg_len_of_snake, max_len_of_snake

def learn(model, target_model, memory, optimizer, num_updates, batch_size, device, target_update_frequency):
    total_loss = 0
    MSE = nn.MSELoss()

    if len(memory) < batch_size:
        return 0.0

    for i in range(num_updates):
        model.reset_noise()
        target_model.reset_noise()

        optimizer.zero_grad()
        sample = memory.sample(batch_size)
        states, actions, rewards, next_states, dones = sample

        states = torch.stack(states).to(device)
        next_states = torch.stack(next_states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)

        q_local = model(states)
        next_q_value = target_model(next_states)

        Q_expected = q_local.gather(1, actions.unsqueeze(1)).squeeze(1)
        Q_targets_next = torch.max(next_q_value, 1)[0] * (1 - dones)
        Q_targets = rewards + 0.9 * Q_targets_next

        loss = MSE(Q_expected, Q_targets)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if i % target_update_frequency == 0:
            target_model.load_state_dict(model.state_dict())

    return total_loss

def train(args, model, target_model, memory, optimizer, device):
    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []
    avg_len_array = []
    avg_max_len_array = []
    time_start = time.time()

    for i_episode in range(args.num_episodes + 1):
        score, avg_len, max_len = run_episode_parallel(memory,args.num_games, args.num_processes)

        scores_deque.append(score)
        scores_array.append(score)
        avg_len_array.append(avg_len)
        avg_max_len_array.append(max_len)

        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)

        total_loss = learn(model, target_model, memory, optimizer, 
                          args.num_updates, args.batch_size, device, 
                          args.target_update_frequency)

        dt = int(time.time() - time_start)

        if i_episode % 10 == 0 and i_episode > 0:
            print('Ep.: {:6}, Loss: {:.3f}, Avg.Score: {:.2f}, Avg.LenOfSnake: {:.2f}, Max.LenOfSnake: {:.2f} Time: {:02}:{:02}:{:02}'.format(
                i_episode, total_loss, score, avg_len, max_len, dt//3600, dt%3600//60, dt%60))

        memory.truncate()

        if i_episode % 250 == 0 and i_episode > 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"Snake_{i_episode}"))

    return scores_array, avg_scores_array, avg_len_array, avg_max_len_array

def main():
    print("Entering main function...")  # 调试信息

    # 解析参数
    args = parse_args()
    global_gridsize = args.gridsize

    print(f"Creating checkpoint directory: {args.checkpoint_dir}")  # 调试信息
    # 创建检查点目录
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # 设置日志
    log = open(os.path.join(args.checkpoint_dir, "log.txt"), "w+", buffering=1)
    sys.stdout = log
    sys.stderr = log

    # 设置设备
    device = get_device()
    print(f"Using device: {device}")

    # 初始化模型
    print("Initializing models...")  # 调试信息
    model = QCNNNoisyNet(input_channels=4, gridsize=args.gridsize, output_dim=5).to(device)
    model.load_state_dict(torch.load('/Users/zhanghanxing/Desktop/work/abv/Snakeee/dir_chk_6/Snake_5000', weights_only=True))

    target_model = QCNNNoisyNet(input_channels=4, gridsize=args.gridsize, output_dim=5).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    # 初始化其他组件
    print("Initializing other components...")  # 调试信息
    memory = ReplayMemory(1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 训练
    print("Starting training...")  # 调试信息
    scores, avg_scores, avg_len_of_snake, max_len_of_snake = train(
        args, model, target_model, memory, optimizer, device)

    print("Training completed. Plotting results...")  # 调试信息
    # ... [绘图代码保持不变] ...
if __name__ == '__main__':
    print("Script started")  # 调试信息
    try:
        mp.freeze_support()
        print("Multiprocessing support initialized")  # 调试信息
        main()
    except Exception as e:
        print(f"Error occurred: {e}")  # 错误处理
        raise