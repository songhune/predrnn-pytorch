import os
import torch
from torchsummary import summary as torch_summary
from torch.optim import Adam
from core.models import predrnn, predrnn_v2, action_cond_predrnn, action_cond_predrnn_v2, predrnn_localbi
class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            'predrnn': predrnn.BiDirectionalRNN,
            'predrnn_v2': predrnn_v2.RNN,
            'action_cond_predrnn': action_cond_predrnn.RNN,
            'action_cond_predrnn_v2': action_cond_predrnn_v2.RNN,
            
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt'+'-'+str(itr))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path)
        self.network.load_state_dict(stats['net_param'])

    def train(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        self.optimizer.zero_grad()
        next_frames, loss = self.network(frames_tensor, mask_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def test(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames, _ = self.network(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy()
    
    def summary(self):
        print("Model structure:")
        print(self.network)
        
        # configs에서 필요한 정보 추출
        batch_size = self.configs.batch_size
        total_length = self.configs.total_length
        img_channel = self.configs.img_channel
        img_width = self.configs.img_width
        img_height = getattr(self.configs, 'img_height', img_width)

        # 입력 shape 계산 (채널을 마지막으로 이동)
        input_shape = (batch_size, total_length, img_height, img_width, img_channel)
        
        print(f"Input shape (B, T, H, W, C): {input_shape}")
        
        summary_str = ""
        def summary_hook(module, input, output):
            nonlocal summary_str
            summary_str += f"\n{module.__class__.__name__}"
            summary_str += f"\n\tInput shape: {[tuple(i.shape) for i in input if isinstance(i, torch.Tensor)]}"
            summary_str += f"\n\tOutput shape: {output[0].shape if isinstance(output, tuple) else output.shape}"
            summary_str += f"\n\tNum parameters: {sum(p.numel() for p in module.parameters())}"

        hooks = []
        for name, module in self.network.named_modules():
            if not list(module.children()):  # 마지막 레이어만
                hooks.append(module.register_forward_hook(summary_hook))

        # 더미 입력으로 네트워크 실행
        dummy_input = torch.zeros(input_shape, device=self.configs.device)
        dummy_mask = torch.zeros((batch_size, total_length-1, img_height, img_width, 1), device=self.configs.device)

        try:
            self.network(dummy_input, dummy_mask)
        except Exception as e:
            print(f"Error during forward pass: {e}")
            print("This error might be due to specific requirements of the model's forward method.")
            print("Please check the model's forward method implementation.")

        for hook in hooks:
            hook.remove()

        print(summary_str)
        print(f"\nTotal parameters: {sum(p.numel() for p in self.network.parameters())}")