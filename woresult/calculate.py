import torch
import torch.nn as nn
import torchvision.transforms as transforms
#from torch.utils.data import DataLoader
from lpips import LPIPS
from skimage.metrics import structural_similarity as ssim
from core.data_provider import datasets_factory
#from piq import psnr, mse
from core.utils import preprocess

import core.trainer as trainer

from core.models.model_factory import Model


# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')

# training/test
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--device', type=str, default='cpu:0')

# data
parser.add_argument('--dataset_name', type=str, default='mnist')
parser.add_argument('--train_data_paths', type=str, default='/data/songhune/data/moving-mnist-example/moving-mnist-train.npz')
parser.add_argument('--valid_data_paths', type=str, default='/data/songhune/data/moving-mnist-example/moving-mnist-valid.npz')
parser.add_argument('--save_dir', type=str, default='/data/songhune/checkpoints/mnist_predrnn')
parser.add_argument('--gen_frm_dir', type=str, default='/data/songhune/results/mnist_predrnn')
parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--total_length', type=int, default=20)
parser.add_argument('--img_width', type=int, default=64)
parser.add_argument('--img_channel', type=int, default=1)

# model
parser.add_argument('--model_name', type=str, default='predrnn')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='64,64,64,64')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--layer_norm', type=int, default=1)
parser.add_argument('--decouple_beta', type=float, default=0.1)

# reverse scheduled sampling
parser.add_argument('--reverse_scheduled_sampling', type=int, default=0)
parser.add_argument('--r_sampling_step_1', type=float, default=25000)
parser.add_argument('--r_sampling_step_2', type=int, default=50000)
parser.add_argument('--r_exp_alpha', type=int, default=5000)
# scheduled sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=50000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

# optimization
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--reverse_input', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_iterations', type=int, default=80000)
parser.add_argument('--display_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=5000)
parser.add_argument('--snapshot_interval', type=int, default=5000)
parser.add_argument('--num_save_samples', type=int, default=10)
parser.add_argument('--n_gpu', type=int, default=1)

# visualization of memory decoupling
parser.add_argument('--visual', type=int, default=0)
parser.add_argument('--visual_path', type=str, default='./decoupling_visual')

# action-based predrnn
parser.add_argument('--injection_action', type=str, default='concat')
parser.add_argument('--conv_on_input', type=int, default=0, help='conv on input')
parser.add_argument('--res_on_conv', type=int, default=0, help='res on conv')
parser.add_argument('--num_action_ch', type=int, default=4, help='num action ch')

args = parser.parse_args()
def test_wrapper(model):
    model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(
        args.dataset_name, args.train_data_paths, args.valid_data_paths, args.batch_size, args.img_width,
        seq_length=args.total_length, injection_action=args.injection_action, is_training=False)
    trainer.test(model, test_input_handle, args, 'test_result')
    
with torch.no_grad():
    for data in test_loader:
        input, target = data  # 데이터셋의 입력과 타겟을 적절히 분리
        output = model(input)

        # MSE 계산
        mse_value = mse(output, target)
        mse_list.append(mse_value.item())

        # SSIM 계산
        ssim_value = ssim(output.squeeze().cpu().numpy(), target.squeeze().cpu().numpy(), multichannel=True)
        ssim_list.append(ssim_value)

        # PSNR 계산
        psnr_value = psnr(output, target, data_range=1.0)  # 데이터 범위에 맞춰 조정 필요
        psnr_list.append(psnr_value.item())

        # LPIPS 계산
        lpips_value = loss_fn_lpips(output, target)
        lpips_list.append(lpips_value.item())

# 모델 로드
model = Model()  # 실제 모델 클래스 사용
checkpoint = torch.load('model_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# LPIPS 초기화
loss_fn_lpips = LPIPS(net='alex')

mse_list = []
ssim_list = []
psnr_list = []
lpips_list = []
# 평균 결과 출력
print(f'Average MSE: {sum(mse_list) / len(mse_list)}')
print(f'Average SSIM: {sum(ssim_list) / len(ssim_list)}')
print(f'Average PSNR: {sum(psnr_list) / len(psnr_list)}')
print(f'Average LPIPS: {sum(lpips_list) / len(lpips_list)}')
