__author__ = 'yunbo'
__editor__ = 'songhune'
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.layers.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell

class OpticalFlowEstimator(nn.Module):
    def __init__(self):
        super(OpticalFlowEstimator, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)
    
    
class BiDirectionalRNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(BiDirectionalRNN, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        # for local window
        self.min_window_size = configs.min_window_size
        self.max_window_size = configs.max_window_size
        self.optical_flow_estimator = OpticalFlowEstimator()
        self.ema_alpha = 0.8  # Exponential Moving Average의 알파 값
        
        self.prev_window_size = (self.min_window_size + self.max_window_size) // 2  # 초기 window_size
        
        cell_list = []

        width = configs.img_width // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
    # 순방향, 역방향 LSTM 셀 (채널 수 변화 없음)
        self.forward_cells = nn.ModuleList([
            SpatioTemporalLSTMCell(self.frame_channel if i == 0 else self.num_hidden[i-1],
                                self.num_hidden[i], width, configs.filter_size,
                                configs.stride, configs.layer_norm)
            for i in range(num_layers)
        ])
        self.backward_cells = nn.ModuleList([
            SpatioTemporalLSTMCell(self.frame_channel if i == 0 else self.num_hidden[i-1],
                                self.num_hidden[i], width, configs.filter_size,
                                configs.stride, configs.layer_norm)
            for i in range(num_layers)
        ])
                # 최종 출력 레이어 (채널 수를 원래대로 줄임)
        self.conv_last = nn.Conv2d(self.num_hidden[num_layers - 1] * 2, self.frame_channel,
                           kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, frames_tensor, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)

        h_t_forward = [torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device) for i in range(self.num_layers)]
        c_t_forward = [torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device) for i in range(self.num_layers)]
        h_t_backward = [torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device) for i in range(self.num_layers)]
        c_t_backward = [torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device) for i in range(self.num_layers)]
        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)

        for t in range(self.configs.total_length - 1):
            # 동적 window_size 계산
            window_size = self._compute_dynamic_window_size(frames[:, max(0, t-1):t+2])
            # 현재 시점을 중심으로 한 윈도우 계산
            start = max(0, t - window_size // 2)
            end = min(self.configs.total_length - 1, t + window_size // 2)
            
            # Forward pass
            for i in range(start, t+1):
                net = self._get_input(frames, mask_true, x_gen, i)
                h_t_forward[0], c_t_forward[0], memory = self.cell_list[0](net, h_t_forward[0], c_t_forward[0], memory, direction='forward')
                for j in range(1, self.num_layers):
                    h_t_forward[j], c_t_forward[j], memory = self.cell_list[j](h_t_forward[j-1], h_t_forward[j], c_t_forward[j], memory, direction='forward')

            # Backward pass
            for i in range(end, t, -1):
                net = self._get_input(frames, mask_true, x_gen, i)
                h_t_backward[0], c_t_backward[0], memory = self.cell_list[0](net, h_t_backward[0], c_t_backward[0], memory, direction='backward')
                for j in range(1, self.num_layers):
                    h_t_backward[j], c_t_backward[j], memory = self.cell_list[j](h_t_backward[j-1], h_t_backward[j], c_t_backward[j], memory, direction='backward')

            # Combine forward and backward hidden states
            h_combined = [torch.cat([h_f, h_b], dim=1) for h_f, h_b in zip(h_t_forward, h_t_backward)]
            
            x_gen = self.conv_last(h_combined[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
            
        loss = self.MSE_criterion(next_frames, frames_tensor[:, self.configs.input_length:])
        return next_frames, loss

    def _forward_pass(self, frames, mask_true):
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        h_t = []
        c_t = []
        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)

        forward_h = []
        forward_c = []
        forward_m = []

        for t in range(self.configs.total_length - self.configs.input_length):
            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.forward_cells[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.forward_cells[i](h_t[i - 1], h_t[i], c_t[i], memory)
                print(f"Forward pass - Layer {i}, Time step {t}: h_t shape = {h_t[i].shape}")


            x_gen = self.conv_last(h_t[self.num_layers - 1])
            forward_h.append(h_t[-1])
            forward_c.append(c_t[-1])
            forward_m.append(memory)

        return forward_h, forward_c, forward_m

    def _backward_pass(self, frames, mask_true):
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        h_t = []
        c_t = []
        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)

        backward_h = []
        backward_c = []
        backward_m = []

        x_gen = frames[:, -1]  # Initialize x_gen with the last frame

        for t in range(self.configs.total_length - 1, self.configs.input_length - 1, -1):
            if t >= self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                    (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.backward_cells[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.backward_cells[i](h_t[i - 1], h_t[i], c_t[i], memory)
                print(f"Backward pass - Layer {i}, Time step {t}: h_t shape = {h_t[i].shape}")

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            backward_h.insert(0, h_t[-1])
            backward_c.insert(0, c_t[-1])
            backward_m.insert(0, memory)

        return backward_h, backward_c, backward_m
    def _get_input(self, frames, mask_true, x_gen, t):
        if self.configs.reverse_scheduled_sampling == 1:
            if t == 0:
                net = frames[:, t]
            else:
                net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
        else:
            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                        (1 - mask_true[:, t - self.configs.input_length]) * x_gen
        
        return net
    def _compute_dynamic_window_size(self, frames):
        # 옵티컬 플로우 계산
        flow = self.optical_flow_estimator(torch.cat([frames[:, 0], frames[:, -1]], dim=1))
        flow_magnitude = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2)
        motion_score = flow_magnitude.mean()

        # motion_score에 따라 window_size 결정
        raw_window_size = int(self.min_window_size + 
                              (self.max_window_size - self.min_window_size) * 
                              torch.sigmoid(motion_score))

        # Exponential Moving Average를 사용한 스무딩
        smoothed_window_size = int(self.ema_alpha * self.prev_window_size + (1 - self.ema_alpha) * raw_window_size)
        
        # 윈도우 크기를 홀수로 만들기
        smoothed_window_size = smoothed_window_size if smoothed_window_size % 2 != 0 else smoothed_window_size + 1
        
        # 이전 window_size 업데이트
        self.prev_window_size = smoothed_window_size

        return smoothed_window_size