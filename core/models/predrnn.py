__author__ = 'yunbo'

import torch
import torch.nn as nn
from core.layers.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell


class BiDirectionalRNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(BiDirectionalRNN, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
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

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)

        for t in range(self.configs.total_length - 1):
            # 순방향 LSTM
            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                    (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.forward_cells[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.forward_cells[i](h_t[i - 1], h_t[i], c_t[i], memory)

            forward_h = h_t[-1]

            # 역방향 LSTM (마지막 프레임부터 시작)
            if t >= self.configs.input_length - 1:
                backward_net = frames[:, -t-1]
                backward_h, backward_c, backward_memory = self.backward_cells[0](backward_net, h_t[0], c_t[0], memory)

                for i in range(1, self.num_layers):
                    backward_h, backward_c, backward_memory = self.backward_cells[i](backward_h, h_t[i], c_t[i], backward_memory)

                # 순방향과 역방향 결과 결합
                combined_h = torch.cat([forward_h, backward_h], dim=1)
                x_gen = self.conv_last(combined_h)

                if t >= self.configs.input_length:
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

        for t in range(self.configs.total_length - 1):
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