import torch
import torchvision.transforms as transforms

import cfg
from utils import *

# set your own configs
args = cfg.parse_args()
GPUdevice = torch.device('cuda', args.gpu_device)

# load the original SAM model 
net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
net.eval()

sam_weights = 'checkpoint/sam/sam_vit_b_01ec64.pth'    # load the original SAM weight
with open(sam_weights, "rb") as f:
    state_dict = torch.load(f)
    new_state_dict = {k: v for k, v in state_dict.items() if k in net.state_dict() and net.state_dict()[k].shape == v.shape}
    net.load_state_dict(new_state_dict, strict = False)

# load task-specific adapter
adapter_path = 'OpticCup_Fundus_SAM_1024.pth'
checkpoint_file = os.path.join(adapter_path)
assert os.path.exists(checkpoint_file)
loc = 'cuda:{}'.format(args.gpu_device)
checkpoint = torch.load(checkpoint_file, map_location=loc)

state_dict = checkpoint['state_dict']
if args.distributed != 'none':
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        name = 'module.' + k
        new_state_dict[name] = v
    # load params
else:
    new_state_dict = state_dict

net.load_state_dict(new_state_dict,strict = False)
