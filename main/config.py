import os,sys
import torch
from numpy import random

base_dir = os.path.abspath(os.path.dirname(__file__))

print(base_dir)
sys.path.append(base_dir)

root_dir = base_dir+"/../data"
log_root = base_dir+"/../logs"
res_dir= base_dir+"/../result"
w2v_dir = base_dir+"/../w2v"

embedding_matrix_path = vocab_path = os.path.join(w2v_dir, "embedding_matrix")
train_data_path = os.path.join(root_dir, "chunked/train_*")
eval_data_path = os.path.join(root_dir, "val.bin")
decode_data_path = os.path.join(root_dir, "val.bin")
vocab_path = os.path.join(root_dir, "vocab")
test_path = os.path.join(root_dir, 'AutoMaster_TestSet2_clean.csv')
user_dict = os.path.join(root_dir, 'user_dict.txt')
model_path = os.path.join(log_root, "model/best_model")
save_path = os.path.join(res_dir, 'result.csv')


# 隐藏层
hidden_dim = 256
emb_dim = 300
batch_size = 128
# 最大编码长度
max_enc_steps = 300
# 最大解码长度
max_dec_steps = 80
beam_size = 5
# min_dec_steps = 35
min_dec_steps = 20
vocab_size = 32000


adam_lr = 0.001  # 使用Adam时候的学习率  再小一点


max_grad_norm = 1

pointer_gen = True

is_coverage = True

cov_loss_wt = 1.0

eps = 1e-8
max_iterations = 5000
iter_loop= 100
w2v_iter = 100

# 使用GPU相关
use_gpu = True
GPU = "cuda:0"
USE_CUDA = use_gpu and torch.cuda.is_available()  # 是否使用GPU
NUM_CUDA = torch.cuda.device_count()
DEVICE = torch.device(GPU if USE_CUDA else 'cpu')

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed_all(SEED)
