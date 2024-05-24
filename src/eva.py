import os
import argparse
import sys

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
sys.path.append(current_directory)

import torch

from manager import DLManager

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from utils.config import get_cfg

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='../configs/config_mvsec.yaml')
parser.add_argument('--data_root', type=str, default='../dataset')
parser.add_argument('--checkpoint_path', type=str, default='../checkpoint/mvsec_checkpoint.pth')
parser.add_argument('--save_root', type=str, default='../test')
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--cuda_id', type=str,default='0')

args = parser.parse_args()

args.is_distributed = False
args.is_master = True
args.world_size = 1
args.local_rank = 0

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
num_gpus = torch.cuda.device_count()
print('Device: ', num_gpus)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

assert os.path.isdir(args.data_root)
cfg = get_cfg(args.config_path)
exp_manager = DLManager(device, args, cfg)

print(args.checkpoint_path)

exp_manager.eva()
