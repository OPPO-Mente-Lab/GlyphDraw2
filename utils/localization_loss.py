# Copyright (c) OPPO Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import re
import torch.nn as nn
import types
import torch.nn.functional as F
import torch
from third_party.ip_adapter.attention_processor import CAAttnProcessor2_0
from third_party.ip_adapter.attention_processor import IPAttnProcessor2_0
from third_party.ip_adapter.attention_processor import IPAttnProcessor
from third_party.ip_adapter.attention_processor import CAAttnProcessor
from third_party.ip_adapter.attention_processor import  AttnProcessor_ori,AttnProcessor
from third_party.ip_adapter.attention_processor import CAAttnProcessor2_0_IP
from third_party.ip_adapter.attention_processor import CAAttnProcessor_IP


from diffusers.models.attention_processor import (
    Attention,
    # AttnProcessor,
    AttnProcessor2_0,
)

def unet_store_cross_attention_scores_ori(unet, attention_scores):
    UNET_LAYER_NAMES = [
        "mid_block",
        "up_blocks.1",
        "up_blocks.2",
        "up_blocks.3",
    ]
    def make_new_get_attention_scores_fn(name):
        def new_get_attention_scores(module, query, key, attention_mask=None):
            attention_probs = module.old_get_attention_scores(
                query, key, attention_mask
            )
            # attention_scores[name] = torch.bmm(attention_probs, query)
            attention_scores[name] = attention_probs
            return attention_probs
        return new_get_attention_scores
    for name, module in unet.named_modules():
        if isinstance(module, Attention) and "attn2" in name:
            if not any(layer in name for layer in UNET_LAYER_NAMES):
                continue
            if isinstance(module.processor, AttnProcessor2_0):
                module.set_processor(AttnProcessor())
            module.old_get_attention_scores = module.get_attention_scores
            module.get_attention_scores = types.MethodType(
                make_new_get_attention_scores_fn(name), module
            )
    return unet
def unet_store_cross_attention_scores_ori1(unet, attention_scores):
    UNET_LAYER_NAMES = [
        "mid_block",
        "up_blocks.1",
        "up_blocks.2",
        "up_blocks.3",
    ]
    def make_new_get_attention_scores_fn(name):
        def new_get_attention_scores(module, query, key, attention_mask=None):
            attention_probs = module.old_get_attention_scores1(
                query, key, attention_mask
            )
            # attention_scores[name] = torch.bmm(attention_probs, query)
            attention_scores[name] = attention_probs
            return attention_probs
        return new_get_attention_scores
    for name, module in unet.named_modules():
        if isinstance(module, Attention) and "attn2" in name:
            if not any(layer in name for layer in UNET_LAYER_NAMES):
                continue
            if isinstance(module.processor, AttnProcessor2_0):
                module.set_processor(AttnProcessor())
            module.old_get_attention_scores1 = module.get_attention_scores
            module.get_attention_scores = types.MethodType(
                make_new_get_attention_scores_fn(name), module
            )
    return unet


def unet_store_cross_attention_scores_id(unet, attention_scores):
    UNET_LAYER_NAMES = [
        "mid_block",
        "up_blocks.1",
        "up_blocks.2",
        "up_blocks.3",
    ]
    def make_new_get_attention_scores_fn(name):
        def new_get_attention_scores(module, query, key, attention_mask=None):
            attention_probs = module.old_get_attention_scores1(
                query, key, attention_mask
            )
            # attention_scores[name] = torch.bmm(attention_probs, query)
            attention_scores[name] = attention_probs
            return attention_probs
        return new_get_attention_scores
    for name, module in unet.named_modules():
        if isinstance(module, Attention) and "attn2" in name:
            if not any(layer in name for layer in UNET_LAYER_NAMES):
                continue
            if isinstance(module.processor, IPAttnProcessor2_0):
                cross_attention_dim =  unet.unet.config.cross_attention_dim
                if name.startswith("unet.mid_block"):
                    hidden_size = unet.unet.config.block_out_channels[-1]
                elif name.startswith("unet.up_blocks"):
                    block_id = int(name[len("unet.up_blocks.")])
                    hidden_size = list(reversed(unet.unet.config.block_out_channels))[block_id]
                module.set_processor(IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)) # set_processor 不能用2.0 
            module.old_get_attention_scores1 = module.get_attention_scores
            module.get_attention_scores = types.MethodType(
                make_new_get_attention_scores_fn(name), module
            )
    return unet


def unet_store_cross_attention_scores_id_ca(unet, attention_scores):

    UNET_LAYER_NAMES = [
        "mid_block",
        "up_blocks.1",
        "up_blocks.2",
        "up_blocks.3",
    ]
    def make_new_get_attention_scores_fn(name):
        def new_get_attention_scores(module, query, key, attention_mask=None):
            attention_probs = module.old_get_attention_scores(
                query, key, attention_mask
            )
            attention_scores[name] = attention_probs
            return attention_probs

        return new_get_attention_scores

    for name, module in unet.named_modules():
        if isinstance(module, Attention) and "attn2" in name:
            if not any(layer in name for layer in UNET_LAYER_NAMES):
                continue
            if isinstance(module.processor, CAAttnProcessor2_0_IP):
                cross_attention_dim =  unet.unet.config.cross_attention_dim
                layer_name = name.split(".processor")[0]
                layer_d = "".join(re.findall("\d+", layer_name)[:-1])
                if name.startswith("unet.mid_block"):
                    hidden_size = unet.unet.config.block_out_channels[-1]
                elif name.startswith("unet.up_blocks"):
                    if layer_d[:2] in ["00","10"]: ## 由于非对称结构，忽略up1和up2第一层block快
                        continue
                    else:
                        block_id = int(name[len("unet.up_blocks.")])
                        hidden_size = list(reversed(unet.unet.config.block_out_channels))[block_id]                    
                module.set_processor(CAAttnProcessor_IP(hidden_size=hidden_size,layer_name=layer_d, cross_attention_dim=cross_attention_dim)) # set_processor 不能用2.0 
            module.old_get_attention_scores = module.get_attention_scores
            module.get_attention_scores = types.MethodType(
                make_new_get_attention_scores_fn(name), module
            )

    return unet

def unet_store_cross_attention_scores(unet, attention_scores):

    UNET_LAYER_NAMES = [
        "mid_block",
        "up_blocks.1",
        "up_blocks.2",
        "up_blocks.3",
    ]
    def make_new_get_attention_scores_fn(name):
        def new_get_attention_scores(module, query, key, attention_mask=None):
            attention_probs = module.old_get_attention_scores(
                query, key, attention_mask
            )
            attention_scores[name] = attention_probs
            return attention_probs

        return new_get_attention_scores

    for name, module in unet.named_modules():
        if isinstance(module, Attention) and "attn2" in name:
            if not any(layer in name for layer in UNET_LAYER_NAMES):
                continue
            if isinstance(module.processor, CAAttnProcessor2_0):
                cross_attention_dim =  unet.config.cross_attention_dim
                layer_name = name.split(".processor")[0]
                layer_d = "".join(re.findall("\d+", layer_name)[:-1])
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    if layer_d[:2] in ["00","10"]: ## 由于非对称结构，忽略up1和up2第一层block快
                        continue
                    else:
                        block_id = int(name[len("up_blocks.")])
                        hidden_size = list(reversed(unet.config.block_out_channels))[block_id]                    
                module.set_processor(CAAttnProcessor(hidden_size=hidden_size,layer_name=layer_d, cross_attention_dim=cross_attention_dim)) # set_processor 不能用2.0 
            module.old_get_attention_scores = module.get_attention_scores
            module.get_attention_scores = types.MethodType(
                make_new_get_attention_scores_fn(name), module
            )

    return unet

def unet_gligen_store_cross_attention_scores(unet, attention_scores, layers=5):
    from diffusers.models.attention_processor import (
        Attention,
        AttnProcessor,
        AttnProcessor2_0,
    )

    UNET_LAYER_NAMES = [
        "down_blocks.0",
        "down_blocks.1",
        "down_blocks.2",
        "mid_block",
        "up_blocks.1",
        "up_blocks.2",
        "up_blocks.3",
    ]

    start_layer = (len(UNET_LAYER_NAMES) - layers) // 2
    end_layer = start_layer + layers
    applicable_layers = UNET_LAYER_NAMES[start_layer:end_layer]

    def make_new_get_attention_scores_fn(name):
        def new_get_attention_scores(module, query, key, attention_mask=None):
            attention_probs = module.old_get_attention_scores(query, key, attention_mask)
            attention_scores[name] = attention_probs
            return attention_probs

        return new_get_attention_scores

    for name, module in unet.named_modules():
        if isinstance(module, Attention) and "attn2" in name:
            if not any(layer in name for layer in applicable_layers):
                continue
            if isinstance(module.processor, AttnProcessor2_0):
                module.set_processor(AttnProcessor())
            module.old_get_attention_scores = module.get_attention_scores
            module.get_attention_scores = types.MethodType(make_new_get_attention_scores_fn(name), module)

    return unet


def get_object_localization_loss_for_one_layer(
    cross_attention_scores,
    object_segmaps,
    object_token_idx,
    object_token_idx_mask,
    loss_fn,
):
    bxh, num_noise_latents, num_text_tokens = cross_attention_scores.shape
    b, _, h, w = object_segmaps.shape
    max_num_objects = object_token_idx_mask.shape[-1]
    
    x,y=h,w
    while h>0:
        w,h=h,w%h
    x=int(x/w)
    y=int(y/w)

    size_h = int((num_noise_latents/(x*y))**0.5)*x
    size_w = int((num_noise_latents/(x*y))**0.5)*y

    # Resize the object segmentation maps to the size of the cross attention scores
    object_segmaps = F.interpolate(
        object_segmaps, size=(size_w, size_h), mode="bilinear", antialias=True
    )  # (b, max_num_objects, size, size)

    object_segmaps = object_segmaps.view(
        b, max_num_objects, -1
    )  # (b, max_num_objects, num_noise_latents)

    num_heads = bxh // b

    cross_attention_scores = cross_attention_scores.view(
        b, num_heads, num_noise_latents, num_text_tokens
    )
    # object_token_idx = torch.cat((object_token_idx,object_token_idx+77),dim=1)
    # object_token_idx_mask = object_token_idx_mask.repeat(1,2)

    # Gather object_token_attn_prob
    object_token_attn_prob = torch.gather(
        cross_attention_scores,
        dim=3,
        index=object_token_idx.view(b, 1, 1, max_num_objects).expand(
            b, num_heads, num_noise_latents, max_num_objects
        ),
    )  # (b, num_heads, num_noise_latents, max_num_objects)

    object_segmaps = (
        object_segmaps.permute(0, 2, 1)
        .unsqueeze(1)
        .expand(b, num_heads, num_noise_latents, max_num_objects)
    )

    loss = loss_fn(object_token_attn_prob, object_segmaps)

    loss = loss * object_token_idx_mask.view(b, 1, max_num_objects)
    object_token_cnt = object_token_idx_mask.sum(dim=1).view(b, 1) + 1e-5
    loss = (loss.sum(dim=2) / object_token_cnt).mean()

    return loss


def get_object_localization_loss(
    cross_attention_scores,
    object_segmaps,
    image_token_idx,
    image_token_idx_mask,
    loss_fn,
):
    num_layers = len(cross_attention_scores)
    loss = 0
    for k, v in cross_attention_scores.items():
        layer_loss = get_object_localization_loss_for_one_layer(
            v, object_segmaps, image_token_idx, image_token_idx_mask, loss_fn
        )
        loss += layer_loss
    return loss / num_layers


class BalancedL1Loss(nn.Module):
    def __init__(self, threshold=1.0, normalize=False):
        super().__init__()
        self.threshold = threshold
        self.normalize = normalize

    def forward(self, object_token_attn_prob, object_segmaps):
        if self.normalize:
            object_token_attn_prob = object_token_attn_prob / (
                object_token_attn_prob.max(dim=2, keepdim=True)[0] + 1e-5
            )
        background_segmaps = 1 - object_segmaps
        background_segmaps_sum = background_segmaps.sum(dim=2) + 1e-5
        object_segmaps_sum = object_segmaps.sum(dim=2) + 1e-5

        background_loss = (object_token_attn_prob * background_segmaps).sum(
            dim=2
        ) / background_segmaps_sum

        object_loss = (object_token_attn_prob * object_segmaps).sum(
            dim=2
        ) / object_segmaps_sum

        return background_loss - object_loss
        # return F.mse_loss(background_loss,object_loss)
