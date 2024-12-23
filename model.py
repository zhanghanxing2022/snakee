import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from NoisyNet.hardware_model import add_noise_calculate_power

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4, args=None):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.args = args

        # 确保所有参数都是 float32 类型
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).float())
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features).float())
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features).float())

        self.bias_mu = nn.Parameter(torch.empty(out_features).float())
        self.bias_sigma = nn.Parameter(torch.empty(out_features).float())
        self.register_buffer('bias_epsilon', torch.empty(out_features).float())

        self.reset_parameters()
        self.reset_noise()

        # 指定使用哪个指标
        self.metric_type = args.metric_type if hasattr(args, 'metric_type') else 'power'
        # 指定使用移动平均还是当前值
        self.use_moving_avg = args.use_moving_avg if hasattr(args, 'use_moving_avg') else False
        # 指标阈值
        self.metric_threshold = args.metric_threshold if hasattr(args, 'metric_threshold') else None

        # 用于存储指标
        self.max_history = 1000
        self.power = []
        self.nsr = []
        self.input_sparsity = []
        self.moving_avg_power = 0
        self.moving_avg_nsr = 0
        self.moving_avg_sparsity = 0
        self.beta = args.beta if hasattr(args, 'beta') else 0.99
    def get_current_metric(self):
        """获取当前正在使用的指标值"""
        if self.use_moving_avg:
            if self.metric_type == 'power':
                return self.moving_avg_power
            elif self.metric_type == 'nsr':
                return self.moving_avg_nsr
            elif self.metric_type == 'sparsity':
                return self.moving_avg_sparsity
        else:
            if self.metric_type == 'power':
                return self.power[-1] if self.power else 0
            elif self.metric_type == 'nsr':
                return self.nsr[-1] if self.nsr else 0
            elif self.metric_type == 'sparsity':
                return self.input_sparsity[-1] if self.input_sparsity else 0
        return 0
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def update_metrics(self, power=None, nsr=None, sparsity=None):
        """更新指标，保持历史记录在限定长度内"""
        if power is not None:
            self.power.append(power)
            if len(self.power) > self.max_history:
                self.power.pop(0)
            # 更新移动平均
            self.moving_avg_power = self.beta * self.moving_avg_power + (1 - self.beta) * power

        if nsr is not None:
            self.nsr.append(nsr)
            if len(self.nsr) > self.max_history:
                self.nsr.pop(0)
            self.moving_avg_nsr = self.beta * self.moving_avg_nsr + (1 - self.beta) * nsr

        if sparsity is not None:
            self.input_sparsity.append(sparsity)
            if len(self.input_sparsity) > self.max_history:
                self.input_sparsity.pop(0)
            self.moving_avg_sparsity = self.beta * self.moving_avg_sparsity + (1 - self.beta) * sparsity

    def get_metrics(self):
        """获取当前指标统计信息"""
        metrics = {
            'power': {
                'current': self.power[-1] if self.power else 0,
                'moving_avg': self.moving_avg_power,
                'min': min(self.power) if self.power else 0,
                'max': max(self.power) if self.power else 0,
                'avg': sum(self.power) / len(self.power) if self.power else 0
            },
            'nsr': {
                'current': self.nsr[-1] if self.nsr else 0,
                'moving_avg': self.moving_avg_nsr,
                'min': min(self.nsr) if self.nsr else 0,
                'max': max(self.nsr) if self.nsr else 0,
                'avg': sum(self.nsr) / len(self.nsr) if self.nsr else 0
            },
            'sparsity': {
                'current': self.input_sparsity[-1] if self.input_sparsity else 0,
                'moving_avg': self.moving_avg_sparsity,
                'min': min(self.input_sparsity) if self.input_sparsity else 0,
                'max': max(self.input_sparsity) if self.input_sparsity else 0,
                'avg': sum(self.input_sparsity) / len(self.input_sparsity) if self.input_sparsity else 0
            }
        }
        return metrics
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device, dtype=torch.float32)  # 指定 dtype
        return x.sign().mul_(x.abs().sqrt_())
    def forward(self, x):
        # 确保输入是 float32
        x = x.float()

        if self.training:
            # 生成噪声权重
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon

            # 计算输出
            output = F.linear(x, weight, bias)

            # 如果需要额外的噪声注入
            if self.args and hasattr(self.args, 'current1') and hasattr(self.args, 'noise'):
                if self.args.current1 > 0 or self.args.noise > 0:
                    arrays = []
                    output = add_noise_calculate_power(
                        self, 
                        self.args,
                        arrays,
                        x,
                        weight,
                        output,
                        layer_type='linear',
                        i=0,
                        layer_num=0,
                        merged_dac=True
                    )
        else:
            output = F.linear(x, self.weight_mu, self.bias_mu)

        return output.float()  # 确保输出也是 float32

    # def forward(self, x):
    #     x = x.float()

    #     if self.training:
    #         weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
    #         bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
    #         output = F.linear(x, weight, bias)

    #         # 获取当前指标值
    #         current_metric = self.get_current_metric()

    #         # 根据指标值调整噪声
    #         if self.metric_threshold is not None:
    #             if current_metric > self.metric_threshold:
    #                 # 可以根据需要调整噪声强度
    #                 self.weight_sigma.data *= 0.95  # 降低噪声
    #             elif current_metric < self.metric_threshold * 0.8:  # 添加一个缓冲区
    #                 self.weight_sigma.data *= 1.05  # 增加噪声

    #         if self.args and hasattr(self.args, 'current1') and hasattr(self.args, 'noise'):
    #             if self.args.current1 > 0 or self.args.noise > 0:
    #                 arrays = []
    #                 output = add_noise_calculate_power(
    #                     self, 
    #                     self.args,
    #                     arrays,
    #                     x,
    #                     weight,
    #                     output,
    #                     layer_type='linear',
    #                     i=0,
    #                     layer_num=0,
    #                     merged_dac=True
    #                 )
    #                 # 更新指标
    #                 if hasattr(self, 'p4'):
    #                     self.update_metrics(
    #                         power=self.p4.item(),
    #                         nsr=torch.mean(torch.abs(arrays[-1][0]) / torch.max(output)).item() if arrays else None,
    #                         sparsity=(x > 0).float().mean().item()
    #                     )
    #     else:
    #         output = F.linear(x, self.weight_mu, self.bias_mu)

    #     return output.float()
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.fc1 = NoisyLinear(args.input_dim, args.hidden_dim, args=args)
        self.fc2 = NoisyLinear(args.hidden_dim, args.hidden_dim, args=args)
        self.fc2 = NoisyLinear(args.hidden_dim, args.hidden_dim, args=args)
        self.fc3 = NoisyLinear(args.hidden_dim, args.output_dim, args=args)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()
class NoisyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NoisyNet, self).__init__()
        self.fc1 = NoisyLinear(input_dim, hidden_dim)
        self.fc2 = NoisyLinear(hidden_dim, hidden_dim)
        self.fc3 = NoisyLinear(hidden_dim, hidden_dim)
        self.fc4 = NoisyLinear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x.float()))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()
        self.fc4.reset_noise()
class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        l1 = self.relu(self.fc1(x.float()))
        l2 = self.relu(self.fc2(l1))
        l3 = self.relu(self.fc3(l2))
        l4 = self.fc4(l3)
        return l4
        
def get_network_input(player, apple):
    proximity = player.getproximity()
    x = torch.cat([torch.from_numpy(player.pos).double(), torch.from_numpy(apple.pos).double(), 
                   torch.from_numpy(player.dir).double(), torch.tensor(proximity).double()])
    return x

def get_network_input2(player, apple):
    gridsize = player.gridsize
    state = np.zeros((4, gridsize, gridsize))# 通道1：蛇头位置
    head_x, head_y = player.pos.astype(int)
    if 0 <= head_x < gridsize and 0 <= head_y < gridsize:
        state[0, head_y, head_x] = 1# 通道2：蛇身位置
    for pos in player.prevpos[:-1]:
        x, y = pos.astype(int)
        if 0 <= x < gridsize and 0 <= y < gridsize:
            state[1, y, x] = 1

    # 通道3：苹果位置
    apple_x, apple_y = apple.pos.astype(int)
    if 0 <= apple_x < gridsize and 0 <= apple_y < gridsize:
        state[2, apple_y, apple_x] = 1

    # 通道4：障碍物探测
    proximity = player.getproximity()
    directions = [(-1,0), (1,0), (0,-1), (0,1)]  # L, R, U, D
    for i, (dx, dy) in enumerate(directions):
        new_x, new_y = head_x + dx, head_y + dy
        # 确保新位置在网格范围内
        if 0 <= new_x < gridsize and 0 <= new_y < gridsize:
            state[3, new_y, new_x] = proximity[i]
    def get_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('cpu')
        return torch.device('cpu')
    device = get_device()
    return torch.FloatTensor(state).to(device)


class QCNNNoisyNet(nn.Module):
    def __init__(self, input_channels=4, gridsize=15, output_dim=5):
        super(QCNNNoisyNet, self).__init__()
       # 使用更小的卷积核和更少的通道
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)

        # 计算卷积后的特征图大小
        # 经过两次stride=2的卷积，特征图大小变为原来的1/4（向上取整）
        conv_output_size = (int(gridsize) + 1) // 4 
        # conv_output_size =  4 
        # 15->8->4
        self.flatten_size = 32 * conv_output_size * conv_output_size

        print(f"Debug - flatten_size: {self.flatten_size}")  # 添加调试信息

        self.fc1 = NoisyLinear(self.flatten_size,128)
        self.fc2 = NoisyLinear(128, output_dim)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        # 添加batch维度如果需要
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        # 打印调试信息
        # print(f"Debug - flattened shape: {x.shape}")

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()