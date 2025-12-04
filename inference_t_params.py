import torch.nn as nn
from keyvitseg_essentials.keyvitseg_original.model import KeySegViT 
from keyvitseg_essentials.UM_Net_loss.UMNet import UMNet as UMNet_loss
from keyvitseg_essentials.UM_Net.UMNet import UMNet
from unet.unet import UNet
from keyvitseg_essentials.UM_Net.UMNet_base import UMNet as UMNet_base
from dynamic_network_architectures.architectures.unet import PlainConvUNet
import time
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis
from thop import profile
import torch.profiler
from torch_flops import TorchFLOPsByFX

random_seed = 42
torch.manual_seed(random_seed)

import time
import torch

import torch
import torch.nn as nn

with torch.no_grad():   
    nnunet = PlainConvUNet(
        input_channels=3,
        num_classes=2,
        n_stages=9,
        features_per_stage=[32, 64, 128, 256, 512, 512, 512, 512, 512],
        conv_op=nn.Conv2d,
        kernel_sizes=[
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3]
        ],
        strides=[
            [1, 1],
            [2, 2],
            [2, 2],
            [2, 2],
            [2, 2],
            [2, 2],
            [2, 2],
            [2, 2],
            [2, 2]
        ],
        n_conv_per_stage=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2, 2, 2, 2],
        conv_bias=True,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={"eps": 1e-05, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True}
    )
        
device = 'cuda' if torch.cuda.is_available() else 'cpu'
models = [
    ("dino_vits16_v1", KeySegViT(model_type='dino_vits16', version='1', stride=8,
    facet='key', checkpoint_choice='None')),
    ("dino_vits14_v2", KeySegViT(model_type='dino_vits14', version='2', stride=7,
    facet='key', checkpoint_choice='None')),
    ("dino_vits14_v2R", KeySegViT(model_type='dino_vits14R', version='2', stride=7, facet='key',
    checkpoint_choice='None')),
    ("dino_vits16_v3", KeySegViT(model_type='dino_vits16', version='3', stride=8,
    facet='key', checkpoint_choice='None')),
    ("umnet_loss", UMNet_loss(num_classes=2)),
    ("umnet", UMNet(num_classes=2)),
    ("umnet_base", UMNet_base(num_classes=2)),
    ("unet", UNet(n_channels=3, n_classes=2, bilinear=True)),
    ("nnunet", nnunet),
]

input_tensor = torch.randn(1, 3, 224, 224, device=device)

warmup_iters = 10
measure_iters = 50

for name, model in models:
    print(f"\n=== Measuring {name} ===")
    model = model.to(device).eval()
    
    if name == "nnunet":
        input_tensor = torch.randn(1, 3, 512, 512).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(input_tensor)
        torch.cuda.synchronize()

    # Timing
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        _ = model(input_tensor)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg = (end - start) / 50
    print(f"{name} inference time: {avg*1000:.2f} ms")

    # ----- Total Parameters ----- #
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {num_params}")




















# ----- FLOPS fcore ----- #
# flops = FlopCountAnalysis(model, input_tensor)
# print(f"FLOPs fcore: {flops.total()}")

# ----- FLOPS thop ----- #
# flops, params = profile(model, inputs=(input_tensor,))
# print(f"FLOPs thop: {flops}, Parameters: {params}")

# ----- FLOPS torch_flops ----- #
# flops_counter = TorchFLOPsByFX(model)
# flops_counter.propagate(input_tensor)
# result_table = flops_counter.print_result_table()
# total_flops = flops_counter.print_total_flops(show=True)
# total_time = flops_counter.print_total_time()
# max_memory = flops_counter.print_max_memory()