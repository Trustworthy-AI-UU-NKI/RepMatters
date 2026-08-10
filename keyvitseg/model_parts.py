# Extractor code from https://github.com/ShirAmir/dino-vit-features -- Copyright (c) 2023, The DINO-ViT Authors. All rights reserved. 
# 
import argparse
import torch
from torch import nn
import torch.nn.modules.utils as nn_utils
import math
import timm
import types
from pathlib import Path
from typing import List, Tuple
import logging
import torch.nn.functional as F
from safetensors.torch import load_file

# -----------------------------------------------------------------------------------------
# Our decoder 
# -----------------------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv without skip connections"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=1, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(out_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Decoder_classic(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.n_classes = n_classes

        self.up1 = Up(1024, 512, self.bilinear)
        self.up2 = Up(512, 256, self.bilinear) # 56
        self.up3 = Up(256, 128, self.bilinear) # 112
        self.up4 = Up(128, 64, bilinear) # 224
        self.outc = OutConv(64, self.n_classes) # 224

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.outc(x)
        logits = F.interpolate(logits, size=(224, 224), mode="bilinear", align_corners=False)
        return logits
    
# -----------------------------------------------------------------------------------------
# ViT extractor 
# -----------------------------------------------------------------------------------------
class ViTExtractor(nn.Module):
    def __init__(self, model_type: str, stride: int, model: nn.Module = None, device: str = 'cuda', version: str = '2', facet: str = 'key'):
        """
        :param model_type: A string specifying the type of model to extract from.
                          [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 |
                          vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]
        :param stride: stride of first convolution layer. small stride -> higher resolution.
        :param model: Optional parameter. The nn.Module to extract from instead of creating a new one in ViTExtractor.
                      should be compatible with model_type.
        """

        super(ViTExtractor, self).__init__()

        self.model_type = model_type
        self.device = device
        if model is not None:
            self.model = model
        else:
            self.model = ViTExtractor.create_model(model_type, version)

        self.model = ViTExtractor.patch_vit_resolution(self.model, stride=stride)
        self.model.to(self.device)
        self.p = self.model.patch_embed.patch_size
        self.stride = self.model.patch_embed.proj.stride

        self.mean = (0.485, 0.456, 0.406) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        self.std = (0.229, 0.224, 0.225) if "dino" in self.model_type else (0.5, 0.5, 0.5)

        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

        self.version = version
        self.facet = facet

    @staticmethod
    def create_model(model_type: str, version: str) -> nn.Module:
        logging.info(f"Creating model {model_type} with version {version}.")
        version = str(version)

        #DINOv3
        if version == '3':
            model = torch.hub.load(
                repo_or_dir="facebookresearch/dinov3",
                model="dinov3_vits16",
                weights="dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
            )
            logging.info(f"Using DINOv3 model {model_type}.")
            
            if model is None:
                raise RuntimeError(f"Failed to load DINOv3 model {model_type}.")
        
        #DINOv2
        elif version == '2':
            if model_type == 'dino_vits14R':
                model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg') # with registers
            
            else:
                model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            logging.info(f"Using DINOv2 model {model_type}.")
            
            if model is None:
                raise RuntimeError(f"Failed to load DINOv2 model {model_type}.")

        #DINOv1
        elif version == '1':   
            if 'dino' in model_type:
                model = torch.hub.load('facebookresearch/dino:main', model_type)
                logging.info(f"Using {model_type} pretrained on imagenet")

        else:  
            logging.info(f"Version of DINO not specified correctly. Using {model_type} from timm library (version 1).")
            temp_model = timm.create_model(model_type, pretrained=True)
            model_type_dict = {
                'vit_small_patch16_224': 'dino_vits16',
                'vit_small_patch8_224': 'dino_vits8',
                'vit_base_patch16_224': 'dino_vitb16',
                'vit_base_patch8_224': 'dino_vitb8'
            }
            model = torch.hub.load('facebookresearch/dino:main', model_type_dict[model_type])
            temp_state_dict = temp_model.state_dict()
            del temp_state_dict['head.weight']
            del temp_state_dict['head.bias']
        return model
            
    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """
        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # example: w: 224, h: 224, patch_size: (14, 14), stride_hw: (7, 7), npatch: 961, N: 1369
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size[1]) // stride_hw[1]
            h0 = 1 + (h - patch_size[0]) // stride_hw[0]
            assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                #size=(int(w0), int(h0)),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False, recompute_scale_factor=False
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod 
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = nn_utils._pair(model.patch_embed.patch_size)
        if stride == patch_size:  # nothing to do
            return model

        stride = nn_utils._pair(stride)
        assert all([(patch_size[i] // stride[i]) * stride[i] == patch_size[i] for i in range(len(patch_size))])
        
        # fix the stride
        model.patch_embed.proj.stride = stride
        
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(ViTExtractor._fix_pos_enc(patch_size, stride), model)
        return model
    
    def _get_hook(self, facet: str):
        """
        Generate a hook method for a specific block and facet.
        """
        if facet in ['attn', 'token']:
            def _hook(model, input, output):
                self._feats.append(output)
            return _hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            self._feats.append(qkv[facet_idx])  # Bxhxtxd
        return _inner_hook
    

    def _register_hooks(self, layers: List[int], facet: str = 'key') -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers: 
                if facet == 'token':
                    # Call the class function explicitly to ensure 'self' and 'facet' are passed
                    hook_fn = ViTExtractor._get_hook(self, facet)
                    self.hook_handlers.append(block.register_forward_hook(hook_fn))
                elif facet == 'attn':
                    # Call the class function explicitly to ensure 'self' and 'facet' are passed
                    hook_fn = ViTExtractor._get_hook(self, facet)
                    self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(hook_fn))
                elif facet in ['key', 'query', 'value']:
                    # Call the class function explicitly to ensure 'self' and 'facet' are passed
                    hook_fn = ViTExtractor._get_hook(self, facet)
                    self.hook_handlers.append(block.attn.register_forward_hook(hook_fn))
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(self, batch: torch.Tensor, layers: List[int] = 11, facet: str = 'key') -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        _, _, H, W = batch.shape
        self._feats = []
        logging.info(f"Facet: {facet}, layers: {layers}")
        self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        # Ensure self.p and self.stride are tuples for correct indexing
        p = self.p if isinstance(self.p, (tuple, list)) else (self.p, self.p)
        stride = self.stride if isinstance(self.stride, (tuple, list)) else (self.stride, self.stride)
        self.num_patches = (1 + (H - p[0]) // stride[0], 1 + (W - p[1]) // stride[1])
        return self._feats
    
    def extract_descriptors(self, batch: torch.Tensor, layer: int = 11, facet: str = 'key',
                            include_cls: bool = False) -> torch.Tensor:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        assert facet in ['key', 'query', 'value', 'token'], f"""{facet} is not a supported facet for descriptors. 
                                                             choose from ['key' | 'query' | 'value' | 'token'] """
        self._extract_features(batch, [layer], facet)
        x = self._feats[0]

        # Keep only spatial tokens
        H, W = self.load_size

        if facet == 'token':
            x.unsqueeze_(dim=1) #Bx1xtxd
        if not include_cls: # mudei
            x = x[:, :, 1:, :]  # remove cls token

        # Ensure self.p and self.stride are tuples for correct indexing
        p = self.p if isinstance(self.p, (tuple, list)) else (self.p, self.p)
        stride = self.stride if isinstance(self.stride, (tuple, list)) else (self.stride, self.stride)

        H_out = 1 + (H - p[0]) // stride[0]
        W_out = 1 + (W - p[1]) // stride[1]
        T_grid = H_out * W_out
        x = x[:, :, -T_grid:, :] # removing the extra tokens, assuming they are also at the beginning
        
        desc = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)  # Bx1xtx(dxh)
        return desc

    def forward(self, batch: torch.Tensor, layer: int = 11, facet: str = 'key',
                include_cls: bool = False) -> torch.Tensor:
        """
        Forward pass through the ViTExtractor.
        :param batch: Input tensor of shape BxCxHxW.
        :param layer: Layer to extract features from. Default is 11.
        :param facet: Facet to extract features from. Default is 'key'.
        :param include_cls: Include the class token in the output. Default is False.
        :return: Extracted descriptors or features.
        """
        return self.extract_descriptors(batch, layer=layer, facet=facet, include_cls=include_cls)

""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def forward(self, batch: torch.Tensor):
    """
    Forward pass through the ViTExtractor.
    :param batch: Input tensor of shape BxCxHxW.
    :return: Extracted descriptors or features.
    """
    descriptors = self.extract_descriptors(batch, layer=11, facet='key')
    return descriptors

