import argparse
import matplotlib.pyplot as plt
import sys
import time
from replay_buffer import ReplayMemory
from collections import deque
from Game import GameEnvironment
from model import QNetwork, get_network_input, NoisyNet, Net
import os
import random
import numpy as np
import torch
import torch.nn as nn
import multiprocessing as mp
from copy import deepcopy
class Args:
    def __init__(self):
        # 网络结构参数
        self.input_dim = 10
        self.hidden_dim = 20
        self.output_dim = 5

def parse_args():
    parser = argparse.ArgumentParser(description='Snake Game DQN Training')
    parser.add_argument('--gridsize', type=int, default=15,
                        help='Size of the game grid')
    parser.add_argument('--num_episodes', type=int, default=1200,
                        help='Number of training episodes')
    parser.add_argument('--target_update_frequency', type=int, default=5,
                        help='How often to update target network')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_updates', type=int, default=20,
                        help='Number of updates per episode')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for training')
    parser.add_argument('--num_games', type=int, default=30,
                        help='Number of games per episode')
    parser.add_argument('--checkpoint_dir', type=str, default='./dir_chk_pall',
                        help='Directory for saving checkpoints')
    return parser.parse_args()
class Config:
    def __init__(self, args):
        # 从命令行参数初始化
        self.gridsize = args.gridsize
        self.num_episodes = args.num_episodes
        self.target_update_frequency = args.target_update_frequency
        self.lr = args.lr
        self.num_updates = args.num_updates
        self.batch_size = args.batch_size
        self.num_games = args.num_games
        self.checkpoint_dir = args.checkpoint_dir

        # 模型参数
        self.input_dim = 10
        self.hidden_dim = 20
        self.output_dim = 5

        # 训练参数
        self.epsilon = 0.1
        self.gamma = 0.9
        self.memory_size = 1000
        self.print_every = 10
        self.save_every = 250
        self.num_processes = 4  # 并行进程数

        # 环境参数
        self.nothing = 0
        self.dead = -1
        self.apple = 1

    def create_dirs(self):
        """创建必要的目录"""
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def setup_logging(self):
        """设置日志"""
        log = open(os.path.join(self.checkpoint_dir, "log.txt"), "w+", buffering=1)
        sys.stdout = log
        sys.stderr = log
        return log


class Trainer:
    def __init__(self, config, model, target_model, optimizer, memory, board):
        self.config = config
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.memory = memory
        self.board = board
        self.MSE = nn.MSELoss()

        # 创建训练统计信息存储
        self.scores_deque = deque(maxlen=100)
        self.scores_array = []
        self.avg_scores_array = []
        self.avg_len_array = []
        self.avg_max_len_array = []


    def learn(self, num_updates, batch_size):
        """训练模型"""
        if len(self.memory) < batch_size:
            return 0.0

        total_loss = 0
        for i in range(num_updates):
            # 重置噪声
            self.model.reset_noise()
            self.target_model.reset_noise()

            self.optimizer.zero_grad()
            states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

            # 转换为张量
            states = torch.cat([x.unsqueeze(0) for x in states], dim=0)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.cat([x.unsqueeze(0) for x in next_states])
            dones = torch.FloatTensor(dones)

            # 计算Q值和损失
            q_local = self.model(states)
            next_q_value = self.target_model(next_states)

            Q_expected = q_local.gather(1, actions.unsqueeze(0).transpose(0, 1)).transpose(0, 1).squeeze(0)
            Q_targets_next = torch.max(next_q_value, 1)[0] * (1 - dones)
            Q_targets = rewards + self.config.gamma * Q_targets_next

            loss = self.MSE(Q_expected, Q_targets)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            if i % self.config.target_update_frequency == 0:
                self.target_model.load_state_dict(self.model.state_dict())

        return total_loss



    def train(self):
        """训练循环"""
        time_start = time.time()
        scores_deque = deque(maxlen=100)
        scores_array = []
        avg_scores_array = []
        avg_len_array = []
        avg_max_len_array = []

        for i_episode in range(self.config.num_episodes + 1):
            # 运行一个episode
            score, avg_len, max_len = self._run_episode()

            # 更新统计信息
            scores_deque.append(score)
            scores_array.append(score)
            avg_len_array.append(avg_len)
            avg_max_len_array.append(max_len)

            avg_score = np.mean(scores_deque)
            avg_scores_array.append(avg_score)

            # 训练模型
            total_loss = self.learn(self.config.num_updates, self.config.batch_size)

            # 打印进度
            dt = int(time.time() - time_start)
            if i_episode % self.config.print_every == 0 and i_episode > 0:
                print('Ep.: {:6}, Loss: {:.3f}, Avg.Score: {:.2f}, Avg.LenOfSnake: {:.2f}, '
                      'Max.LenOfSnake: {:.2f} Time: {:02}:{:02}:{:02}'.format(
                          i_episode, total_loss, score, avg_len, max_len,
                          dt//3600, dt%3600//60, dt%60))

            # 保存检查点
            if i_episode % self.config.save_every == 0 and i_episode > 0:
                torch.save(self.model.state_dict(), 
                         os.path.join(self.config.checkpoint_dir, f"Snake_{i_episode}"))

        return scores_array, avg_scores_array, avg_len_array, avg_max_len_array
    @staticmethod
    def run_single_game(model_state_dict, board, memory):
        """静态方法，可以被多进程调用"""
        model = Net(Args())
        model.load_state_dict(model_state_dict)

        board_copy = deepcopy(board)
        total_reward = 0
        len_of_snake = 0

        while not board_copy.game_over:
            state = get_network_input(board_copy.snake, board_copy.apple)

            with torch.no_grad():
                action_values = model(state)
                action = torch.argmax(action_values).item()

            reward, done, len_of_snake = board_copy.update_boardstate(action)
            next_state = get_network_input(board_copy.snake, board_copy.apple)

            memory.push(state, action, reward, next_state, done)
            total_reward += reward

        return total_reward, len_of_snake

    def _run_episode(self):
        """运行单个episode，使用多进程"""
        model_state_dict = self.model.state_dict()

        with mp.Pool(processes=self.config.num_processes) as pool:
            results = []
            for _ in range(self.config.num_games):
                results.append(pool.apply_async(
                    self.run_single_game, 
                    (model_state_dict, self.board, self.memory)
                ))

            # 收集结果
            rewards = []
            lengths = []
            for r in results:
                reward, length = r.get()
                rewards.append(reward)
                lengths.append(length)

        total_reward = sum(rewards)
        avg_len_of_snake = np.mean(lengths)
        max_len_of_snake = np.max(lengths)

        return total_reward, avg_len_of_snake, max_len_of_snake

    def _print_progress(self, i_episode, total_loss, score, avg_len, max_len, dt):
        """打印训练进度"""
        print('Ep.: {:6}, Loss: {:.3f}, Avg.Score: {:.2f}, Avg.LenOfSnake: {:.2f}, '
              'Max.LenOfSnake: {:.2f} Time: {:02}:{:02}:{:02}'.format(
                  i_episode, total_loss, score, avg_len, max_len,
                  dt//3600, dt%3600//60, dt%60))

    def _save_checkpoint(self, i_episode):
        """保存模型检查点"""
        torch.save(
            self.model.state_dict(),
            os.path.join(self.config.checkpoint_dir, f"Snake_{i_episode}")
        )

def main():
    # 解析参数
    args = parse_args()
    config = Config(args)
    config.create_dirs()
    config.setup_logging()

    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)

    # 初始化组件
    board = GameEnvironment(config.gridsize, nothing=config.nothing, 
                          dead=config.dead, apple=config.apple)
    model = Net(Args())
    target_model = Net(Args())
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    memory = ReplayMemory(config.memory_size)

    # 创建训练器并开始训练
    trainer = Trainer(config, model, target_model, optimizer, memory, board)
    results = trainer.train()

    # 绘制结果

if __name__ == '__main__':
    main()