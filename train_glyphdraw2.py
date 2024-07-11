
# Copyright (c) OPPO Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os,re
import random
import argparse
from easydict import EasyDict as edict
from functools import partial
import types
from tqdm.auto import tqdm
from typing import Callable, List, Optional, Union
from einops import rearrange, repeat,reduce

import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule,Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

from third_party.recognizer import TextRecognizer, create_predictor
from third_party.embedding_manager import get_recog_emb
from utils.localization_loss import unet_store_cross_attention_scores_ori,unet_store_cross_attention_scores_id_ca
from utils.custom_dataset_mul import DataModuleCustom,BUCKETS,MAX_lines
from utils.model_utils import add_module_args,configure_optimizers,get_total_steps
from utils.universal_checkpoint import UniversalCheckpoint

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel
import open_clip

from third_party.ip_adapter.attention_processor import AttnProcessor_ori,CAAttnProcessor2_0_IP
from third_party.ip_adapter.ip_adapter import ImageProjModel
from third_party.ip_adapter.attention_processor import  AttnProcessor2_0 as AttnProcessor

layer_up2_map = {"100":"110","101":"111","110":"120","111":"121"}
layer_up_map = dict(zip([str(i) for i in range(200,220)],[str(i).zfill(3) for i in range(10,30)]))
layer_up_map.update(layer_up2_map)


class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model

    def forward(self, noisy_latents, timesteps, encoder_hidden_states_all,added_cond_kwargs,down_block_additional_residuals,mid_block_additional_residual, glyph_embeds):

        encoder_hidden_fonts = []
        for i in range(len(noisy_latents)):
            text_emb = torch.cat(glyph_embeds[i], dim=0)   
            # text_emb = torch.cat(glyph_embeds[i], dim=0)    
            text_emb = torch.cat([torch.zeros(MAX_lines-len(text_emb),1024).to(noisy_latents.device),text_emb]).half()
            encoder_hidden_fonts.append(text_emb)
        encoder_hidden_fonts = torch.stack(encoder_hidden_fonts) # b*MAX_lines*1024

        ip_tokens = self.image_proj_model(encoder_hidden_fonts)
        encoder_hidden_states_all["ip_tokens"] = ip_tokens
        # Predict the noise residual and compute loss
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states_all,added_cond_kwargs=added_cond_kwargs, \
            down_block_additional_residuals=down_block_additional_residuals,mid_block_additional_residual=mid_block_additional_residual, return_dict=False,
        )[0]
        return noise_pred

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim,out_dim1, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim1)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x2 = self.act_fn(x)
        x2 = self.fc3(x2)
        if self.use_residual:
            x = x + residual
        x1 = torch.mean(x,1)
        return x1,x2

def replace_clip_embeddings(clip_model, text_embs_all, place_holder_token):
    def forward(self, input_ids, token_type_ids=None, position_ids=None
    ) -> torch.Tensor:
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        b, device = input_ids.shape[0], input_ids.device
        for i in range(b):
            idx = input_ids[i] == place_holder_token.to(device)
            if sum(idx) > 0:
                if i >= len(text_embs_all):
                    print('truncation for log images...')
                    break
                text_emb = torch.cat(text_embs_all[i], dim=0)    
                try:
                    words_embeddings[i][idx] = text_emb
                except Exception as e:
                    print(str(e))
                    
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    clip_model.old_forward = clip_model.forward
    clip_model.forward = types.MethodType(forward, clip_model)


def getActivation(activation,name,residuals_present):
    # the hook signature
    if residuals_present:
        def hook(model, input, output):
            activation[name] = output[0]
    else:
        def hook(model, input, output):
            activation[name] = output
    return hook
    
def cast_hook(unet,dicts):
    for i in range(3):
        if i==0:continue 
        for j,attentions in enumerate(unet.down_blocks[i].attentions):
            for k,transformer_blocks in enumerate(attentions.transformer_blocks):
                # transformer_blocks.attn2.register_forward_hook(getActivation(dicts,f'{i}{j}{k}',False))
                transformer_blocks.register_forward_hook(getActivation(dicts,layer_up_map[f'{i}{j}{k}'],False))

    for j,attentions in enumerate(unet.mid_block.attentions):
        for k,transformer_blocks in enumerate(attentions.transformer_blocks):
            transformer_blocks.register_forward_hook(getActivation(dicts,f'{j}{k}',False))


class StableDiffusion(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('OPPO Stable Diffusion Module')
        parser.add_argument('--train_text', default=False)
        parser.add_argument('--train_transformer', default=False)
        parser.add_argument('--local_rank', default=-1, type=int,help='node rank for distributed training')
        return parent_parser

    def __init__(self, args):
        super().__init__()
        self.RESUME_ID = args.load_ckpt_id
        self.RESUME_PATH = args.load_ckpt_path        
        self.align_loss = args.align_loss
        self.controlnet_condition = args.controlnet_condition       
        self.NUMS_SAVE = args.every_n_steps
        self.proj_path = args.proj_path


        self.text_encoder, _, preprocess = open_clip.create_model_and_transforms('xlm-roberta-large-ViT-H-14', pretrained=args.chinese_clip_path)
        self.tokenizer = open_clip.get_tokenizer('xlm-roberta-large-ViT-H-14')
        self.text_encoder.text.output_tokens = True
        self.proj_zh = MLP(1024, 1280, 1024,2048, use_residual=False)
        self.proj_zh.load_state_dict(torch.load(self.proj_path, map_location="cpu"))
 

        self.vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(args.model_path, subfolder="unet")

        self.controlnet = ControlNetModel.from_unet(self.unet, conditioning_channels=1)
        
        self.noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.save_hyperparameters(args)
        self.text_embedding_proj = nn.Linear(2560, 1024)

        rec_model_dir = "./third_party/ocr_weights/ppv3_rec.pth"
        self.text_predictor = create_predictor(rec_model_dir).eval()
        args_ocr = edict()
        args_ocr.rec_image_shape = "3, 48, 320"
        args_ocr.rec_batch_num = 6
        args_ocr.rec_char_dict_path = './third_party/ocr_recog/ppocr_keys_v1.txt'
        args_ocr.use_fp16 = True
        self.cn_recognizer = TextRecognizer(args_ocr, self.text_predictor)
        for param in self.text_predictor.parameters():
            param.requires_grad = False
        self.get_recog_emb = partial(get_recog_emb, self.cn_recognizer)
        
        if self.RESUME_ID:
            controlnet_path = os.path.join(self.RESUME_PATH, f"controlnet_0_{self.RESUME_ID}/pytorch_model.bin")
            text_embedding_proj_path = os.path.join(self.RESUME_PATH, f"text_embedding_proj0_{self.RESUME_ID}/pytorch_model.bin")
            self.controlnet.load_state_dict(torch.load(controlnet_path), strict=True)
            self.text_embedding_proj.load_state_dict(torch.load(text_embedding_proj_path), strict=True)

        self.controlnet_attn2= {}
        cast_hook(self.controlnet,self.controlnet_attn2)

        image_proj_model = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=1024,
            clip_extra_context_tokens=MAX_lines,
        )
        attn_procs = {}
        unet_sd = self.unet.state_dict()
        for name in self.unet.attn_processors.keys():
            if name.endswith("attn1.processor"):
                attn_procs[name] = AttnProcessor()
                continue
            
            cross_attention_dim =  self.unet.config.cross_attention_dim
            layer_name = name.split(".processor")[0]
            layer_d = "".join(re.findall("\d+", layer_name)[:-1])
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
                attn_procs[name] = CAAttnProcessor2_0_IP(hidden_size=hidden_size,layer_name=layer_d, cross_attention_dim=cross_attention_dim)
            elif name.startswith("up_blocks"):
                if layer_d[:2] in ["00","10"]: ## Due to the asymmetric structure, ignoring the fast first layer block of up1 and up2
                    attn_procs[name] = AttnProcessor_ori(hidden_size=hidden_size,layer_name=layer_d, cross_attention_dim=cross_attention_dim)
                else:
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                    attn_procs[name] = CAAttnProcessor2_0_IP(hidden_size=hidden_size,layer_name=layer_d, cross_attention_dim=cross_attention_dim)
                    
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
                attn_procs[name] = AttnProcessor_ori(hidden_size=hidden_size,layer_name=layer_d, cross_attention_dim=cross_attention_dim)
            
        self.unet.set_attn_processor(attn_procs)
        self.ip_adapter = IPAdapter(self.unet, image_proj_model)

        if self.align_loss:
            self.unet_ori = UNet2DConditionModel.from_pretrained(args.model_path, subfolder="unet")
            self.cross_attention_scores = {}
            self.unet_ori = unet_store_cross_attention_scores_ori(self.unet_ori, self.cross_attention_scores)
            self.cross_attention_scores_controlnet = {}
            self.ip_adapter = unet_store_cross_attention_scores_id_ca(self.ip_adapter, self.cross_attention_scores_controlnet)
        
        if self.RESUME_ID:
            unet_path = os.path.join(self.RESUME_PATH, f"unet_0_{self.RESUME_ID}/pytorch_model.bin")
            self.ip_adapter.load_state_dict(torch.load(unet_path), strict=True)

    def setup(self, stage) -> None:
        if stage == 'fit':
            self.total_steps = 999999
            print('Total steps: {}' .format(self.total_steps))

    def configure_optimizers(self):
        params=[]
        names = []
        model_params = []
        for name, p in self.ip_adapter.named_parameters():
            if "image_proj_model" in name or "ip" in name:
                params.append(p) 
                names.append(name)
        total = sum(p.numel() for p in params)
        model_params.append({'params': iter(params)})
        model_params.append({'params': self.controlnet.parameters()})
        model_params.append({'params': self.text_embedding_proj.parameters()})

        return configure_optimizers(self, model_params=model_params)

    def encode_caption_ppocr(self, batch):
        gline_list=[]
        for i in range(len(batch["n_lines"])):
            n_lines = batch["n_lines"][i]
            for j in range(n_lines): 
                gline_list += [batch['gly_line'][i][j]]

        if(len(gline_list) > 0):
            recog_emb = self.get_recog_emb(gline_list)
            enc_glyph = self.text_embedding_proj(recog_emb.reshape(recog_emb.shape[0], -1)) # nums_glyphs * 1024
        self.text_embs_all = []
        n_idx = 0
        for i in range(len(batch['n_lines'])):  # sample index in a batch
            n_lines = batch['n_lines'][i]
            text_embs = []
            for j in range(n_lines):  # line
                text_embs += [enc_glyph[n_idx:n_idx+1]]
                n_idx += 1
            self.text_embs_all += [text_embs]                
        pass

    def encode_caption_bert(self, batch,device):
        glyph = batch["instance_prompt_glyph"]
        self.text_embs_all = []
        for prompt in glyph:
            prompts = prompt.split(" && ")
            input_ids,pinyin_ids = self.tokenizer_ChineseBERT.tokenize_sentence_batch(prompts)
            attention_mask = input_ids.ne(0).type(self.text_encoder_ChineseBERT.embeddings.word_embeddings.weight.dtype).to(device)
            text_embeddings = self.text_encoder_ChineseBERT(input_ids.to(device),pinyin_ids.to(device), attention_mask=attention_mask)
            text_embeddings = self.text_embedding_proj(text_embeddings[1])
            self.text_embs_all += [[text_embeddings]] 

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            self.vae.to(dtype=torch.float32)
            latents = self.vae.encode(batch["pixel_values"].to(dtype=torch.float32)).latent_dist.sample()
            latents = latents.half() * self.vae.config.scaling_factor

        noise = torch.randn(latents.shape).to(latents.device)
        noise = noise.to(dtype=self.unet.dtype)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
 
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        noisy_latents = noisy_latents.to(dtype=self.unet.dtype)
     	
        self.encode_caption_ppocr(batch)

        placeholder_token = self.tokenizer('*')
        placeholder_token = placeholder_token[0,1] 
        replace_clip_embeddings(self.text_encoder, self.text_embs_all, placeholder_token)

        _,encoder_hidden_states = self.text_encoder.encode_text(batch["input_ids"])
        _,encoder_hidden_states_uncond = self.text_encoder.encode_text(batch["input_ids_uncond"])

        add_text_embeds,encoder_hidden_states = self.proj_zh(encoder_hidden_states)
        add_text_embeds_uncond,encoder_hidden_states_uncond = self.proj_zh(encoder_hidden_states_uncond)

        crops_coords_top_left = batch["crops_coords_top_left"]
        original_size = batch["original_size"]
        target_size = torch.tensor([BUCKETS[batch["bucket_id"]]]*len(batch["crops_coords_top_left"]),device=latents.device)
        add_time_ids = torch.cat([original_size,crops_coords_top_left,target_size],1) ##
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        uncond = 0.1
        random = torch.rand(latents.size(0), device=latents.device)
        prompt_mask = rearrange(random < uncond, "n -> n 1 1")
        encoder_hidden_states = torch.where(prompt_mask, encoder_hidden_states_uncond, encoder_hidden_states)
        guided_hint = batch["glyphs"]

        if self.controlnet_condition:
            encoder_hidden_fonts = []
            for i in range(bsz):
                text_emb = torch.cat(self.text_embs_all[i], dim=0)    
                text_emb = torch.cat([torch.zeros(MAX_lines-len(text_emb),1024).to(latents.device),text_emb]).half()
                encoder_hidden_fonts.append(text_emb)
            encoder_hidden_fonts = torch.stack(encoder_hidden_fonts)
            encoder_hidden_fonts = torch.cat([encoder_hidden_fonts,encoder_hidden_fonts],dim=-1)
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_fonts,
                controlnet_cond=guided_hint,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )
        else:
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=guided_hint,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )

        encoder_hidden_states_all = {}
        encoder_hidden_states_all["encoder_hidden_states"] = encoder_hidden_states
        encoder_hidden_states_all["controlnet_attn2"] = self.controlnet_attn2

        model_pred = self.ip_adapter(
            noisy_latents,
            timesteps,
            encoder_hidden_states_all=encoder_hidden_states_all,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            glyph_embeds=self.text_embs_all,
        )

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none")
        loss = loss.mean([1, 2, 3]).mean()
        self.log("train_loss", loss.item(),  on_epoch=False, prog_bar=True, logger=True)
        self.log("lr", lr,  on_epoch=False, prog_bar=True, logger=True)


        if self.align_loss:
            _ = self.unet_ori(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            loss_ca = 0
            num_layers = len(self.cross_attention_scores)
            for v, v_control in zip(self.cross_attention_scores.values(),self.cross_attention_scores_controlnet.values()):
                loss_layer = F.mse_loss(v,v_control)
                loss_ca += loss_layer

            self.log("loss_ca", loss_ca.item(),  on_epoch=False, prog_bar=True, logger=True)

            loss += 0.1*loss_ca

        if self.trainer.global_rank == 0:
            if (self.global_step+1) % self.NUMS_SAVE == 0:
                save_directory = os.path.join(args.default_root_dir,f'controlnet_{self.global_step}')
                os.makedirs(save_directory, exist_ok=True)
                torch.save(self.controlnet.state_dict(), os.path.join(save_directory,"pytorch_model.bin"))

                save_directory = os.path.join(args.default_root_dir,f'text_embedding_proj_{self.global_step}')
                os.makedirs(save_directory, exist_ok=True)
                torch.save(self.text_embedding_proj.state_dict(), os.path.join(save_directory,"pytorch_model.bin"))
                
                save_directory = os.path.join(args.default_root_dir,f'unet_{self.global_step}')
                os.makedirs(save_directory, exist_ok=True)
                torch.save(self.ip_adapter.state_dict(), os.path.join(save_directory,"pytorch_model.bin"))

        return {"loss": loss}

    def on_load_checkpoint(self, checkpoint) -> None:
        global_step_offset = checkpoint["global_step"]
        if 'global_samples' in checkpoint:
            self.consumed_samples = checkpoint['global_samples']
        self.trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser = add_module_args(args_parser)
    args_parser = DataModuleCustom.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = StableDiffusion.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args = args_parser.parse_args()

    model = StableDiffusion(args)
    tokenizer = model.tokenizer

    datamoule = DataModuleCustom(args, tokenizer=tokenizer)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = UniversalCheckpoint(args)

    trainer = Trainer.from_argparse_args(args,callbacks=[lr_monitor,checkpoint_callback])

    trainer.fit(model, datamoule)

