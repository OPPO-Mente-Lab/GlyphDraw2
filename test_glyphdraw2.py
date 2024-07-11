# Copyright (c) OPPO Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os,re,json
import torch
import argparse

from utils import load_config, load_clip, tokenize

from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler, AutoencoderKL, ControlNetModel
from typing import Callable, List, Optional, Union
from diffusers.image_processor import VaeImageProcessor

import torch.nn as nn
import torch.optim as optim


from torchvision.utils import save_image
from torchvision import transforms

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from utils.recognizer import TextRecognizer, create_predictor
from utils.embedding_manager import get_recog_emb

from functools import partial
from easydict import EasyDict as edict
from torchvision import utils as vutils

from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
    LoRAAttnProcessor
)

from test_controlnet_sdxl_utils import *
from ip_adapter.attention_processor import AttnProcessor_ori,CAAttnProcessor2_0_IP
from ip_adapter.attention_processor import CAAttnProcessor2_0 as CAAttnProcessor
from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.attention_processor import  AttnProcessor2_0 as AttnProcessor
from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor
import open_clip

def get_image_paths(data_dir):
    # '/home/notebook/data/group/dengyonglin/AnyText/benchmark/wukong_word/test1k.json'
    image_files = []
    # 读取 JSON 文件
    with open(data_dir, 'r') as f:
        data = json.load(f)

    # 遍历 data_list 中的每个元素
    for item in data['data_list']:
        file = item['img_name']
        image_files.append(file)

    return image_files


class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model

    def forward(self, noisy_latents, timesteps, encoder_hidden_states_all,added_cond_kwargs,down_block_additional_residuals,mid_block_additional_residual, glyph_embeds):

        encoder_hidden_fonts = []
        for i in range(int(len(noisy_latents)/2)):
            text_emb = torch.cat(glyph_embeds[i], dim=0)   
            # text_emb = torch.cat(glyph_embeds[i], dim=0)    
            text_emb = torch.cat([torch.zeros(10-len(text_emb),1024).to(noisy_latents.device),text_emb]).half()
            encoder_hidden_fonts.append(text_emb)
        encoder_hidden_fonts = torch.stack(encoder_hidden_fonts) # b*10*1024

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
        if i==0:continue  ## DownBlock2D object has no attribute 'attentions'
        for j,attentions in enumerate(unet.down_blocks[i].attentions):
            for k,transformer_blocks in enumerate(attentions.transformer_blocks):
                # transformer_blocks.attn2.register_forward_hook(getActivation(dicts,f'{i}{j}{k}',False))
                transformer_blocks.register_forward_hook(getActivation(dicts,layer_up_map[f'{i}{j}{k}'],False))

    for j,attentions in enumerate(unet.mid_block.attentions):
        for k,transformer_blocks in enumerate(attentions.transformer_blocks):
            transformer_blocks.register_forward_hook(getActivation(dicts,f'{j}{k}',False))

layer_up2_map = {"100":"110","101":"111","110":"120","111":"121"}
layer_up_map = dict(zip([str(i) for i in range(200,220)],[str(i).zfill(3) for i in range(10,30)]))
layer_up_map.update(layer_up2_map)


class MLP_plus(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1280, hidden_dim=2048, out_dim1=2048, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )
        self.fc = nn.Linear(out_dim, out_dim1)
        self.use_residual = use_residual
    ## B*77*1024 --> B*1280   B*77*2048  
    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.projector(x)
        x2 = nn.GELU()(x)
        x2 = self.fc(x2)
        if self.use_residual:
            x = x + residual
        x1 = torch.mean(x,1)
        return x1,x2

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class StableDiffusionTest():

    def __init__(self, model_id, text_encoder_path, proj_path, text_proj_path, ctrlnet_path,unet_path,device,fonts):
        super().__init__()
        self.text_encoder, _, preprocess = open_clip.create_model_and_transforms('xlm-roberta-large-ViT-H-14', pretrained=text_encoder_path)
        self.tokenizer = open_clip.get_tokenizer('xlm-roberta-large-ViT-H-14')
        self.text_encoder.text.output_tokens = True

        self.text_encoder = self.text_encoder.to(device)
        self.proj_zh = MLP(1024, 1280, 1024,2048, use_residual=False).to(device).half()
        self.proj_zh.load_state_dict(torch.load(proj_path, map_location="cpu"))

        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler,torch_dtype=torch.float16).to(device)

        # text embedding module的linear层加载
        self.use_gly_line = True
        self.text_embedding_proj = nn.Linear(40*64, 1024).to(device)
        self.text_embedding_proj.load_state_dict(torch.load(text_proj_path, map_location="cpu"))

        #controlnet加载
        # self.controlnet = ControlNetModel.from_unet(self.pipe.unet, conditioning_channels=2).to(device)
        self.controlnet = ControlNetModel.from_unet(self.pipe.unet, conditioning_channels=1).to(device) # guided_hint no concat
        self.controlnet.load_state_dict(torch.load(ctrlnet_path, map_location="cpu"))
        self.controlnet = self.controlnet.half()

        # PPOCR提取特征
        rec_model_dir = "./ocr_weights/ppv3_rec.pth" 
        self.text_predictor = create_predictor(rec_model_dir).to(device).eval()
        args = edict()
        args.rec_image_shape = "3, 48, 320"
        args.rec_batch_num = 6
        args.rec_char_dict_path = './ocr_recog/ppocr_keys_v1.txt'
        args.use_fp16 = False
        self.cn_recognizer = TextRecognizer(args, self.text_predictor)
        for param in self.text_predictor.parameters():
            param.requires_grad = False
        self.get_recog_emb = partial(get_recog_emb, self.cn_recognizer)
        
        # self.pattern = re.compile(r'“(.*?)”')
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(512),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.image_transforms_mask = transforms.Compose(
            [
                transforms.Resize(64, interpolation=transforms.functional._interpolation_modes_from_int(0)),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
            ]
        )


        self.fonts = ImageFont.truetype(fonts, size=60)
        self.device = device
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.pipe.vae_scale_factor)

        self.controlnet_attn2= {}
        cast_hook(self.controlnet,self.controlnet_attn2)


        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=1024,
            clip_extra_context_tokens=10,
        )
        attn_procs = {}
        unet_sd = self.pipe.unet.state_dict()
        for name in self.pipe.unet.attn_processors.keys():
            if name.endswith("attn1.processor"):
                attn_procs[name] = AttnProcessor()
                continue
            
            cross_attention_dim =  self.pipe.unet.config.cross_attention_dim
            layer_name = name.split(".processor")[0]
            layer_d = "".join(re.findall("\d+", layer_name)[:-1])
            # print(f"{layer_name},layer_d:{layer_d}")
            if name.startswith("mid_block"):
                hidden_size = self.pipe.unet.config.block_out_channels[-1]
                attn_procs[name] = CAAttnProcessor2_0_IP(hidden_size=hidden_size,layer_name=layer_d, cross_attention_dim=cross_attention_dim)
            elif name.startswith("up_blocks"):
                if layer_d[:2] in ["00","10"]: ## 由于非对称结构，忽略up1和up2第一层block快
                    attn_procs[name] = AttnProcessor_ori(hidden_size=hidden_size,layer_name=layer_d, cross_attention_dim=cross_attention_dim)
                else:
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.pipe.unet.config.block_out_channels))[block_id]
                    attn_procs[name] = CAAttnProcessor2_0_IP(hidden_size=hidden_size,layer_name=layer_d, cross_attention_dim=cross_attention_dim)
                    
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.pipe.unet.config.block_out_channels[block_id]
                attn_procs[name] = AttnProcessor_ori(hidden_size=hidden_size,layer_name=layer_d, cross_attention_dim=cross_attention_dim)
            
        self.pipe.unet.set_attn_processor(attn_procs)
        self.ip_adapter = IPAdapter(self.pipe.unet, image_proj_model).to(device).half()
        # unet_sd = torch.load(unet_path, map_location="cpu")
        # unet_new = {}
        # for k,v in unet_sd.items():
        #     k_new = "unet."+k
        #     unet_new[k_new] = v
        self.ip_adapter.load_state_dict(torch.load(unet_path, map_location="cpu"))


    def load_guided_hint(self, data_dir, batch_files, device, target_size=(1024, 1024)):
        batch_data = []
        for file_name in batch_files:
            image_path = os.path.join(data_dir, file_name)
            # image =  np.expand_dims(np.array(Image.open(image_path).convert('1')), axis=2).astype(np.float64) * 255
            image =  np.array(Image.open(image_path))
            img_tensor = transforms.ToTensor()(image)
            batch_data.append(img_tensor)

        guided_hint = 1-torch.stack(batch_data).to(device)
        return guided_hint

    def get_guided_hint2(self, new_fonts_list, rect_list, width, height, device): # adaptive generate rects
        batch_size = len(rect_list)
        lgs = []
        for i in range(batch_size):
            new_fonts = new_fonts_list[i]
            rects = rect_list[i]
            all_lg = draw_glyphs(new_fonts, rects, width, height,selected_font)
            # 做透视变换
            # transformed_img = random_perspective(all_lg)
            # lg_tensor = transforms.ToTensor()(transformed_img.astype(np.float64))
            # 不做透视变换
            lg_tensor = transforms.ToTensor()(all_lg.astype(np.uint8)) # 0-1
            filename = ''.join([elem[0] for elem in new_fonts])
            # vutils.save_image(lg_tensor, f'./outputs/test_m_59999/{i}_{filename}.jpg')
            lg_tensor = lg_tensor.to(device)
            lgs+=[lg_tensor]

        glyphs = torch.stack(lgs).to(device)
        guided_hint = 1 - glyphs # no concat
        # vutils.save_image(guided_hint, f'./outputs/test1/guided_hint.jpg')
        return guided_hint

    def encode_caption_glyph(self, fonts):   
        """
        PPOCR提取绘制的字符特征
        """     
        gline_list=[]        
        for i in range(len(fonts)):
            n_lines = len(fonts[i])
            for j in range(n_lines): 
                gline_list += [fonts[i][j]]

        if(len(gline_list) > 0):
            recog_emb = self.get_recog_emb(gline_list)
            enc_glyph = self.text_embedding_proj(recog_emb.reshape(recog_emb.shape[0], -1))
        self.text_embs_all = []
        n_idx = 0
        for i in range(len(fonts)):  # sample index in a batch
            n_lines = len(fonts[i])
            text_embs = []
            for j in range(n_lines):  # line
                text_embs += [enc_glyph[n_idx:n_idx+1]]
                n_idx += 1
            self.text_embs_all += [text_embs] 


    def encode_prompt(self, prompts,fonts, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        # n_lines = self.count_lines(prompts)
        batch_size = len(prompts) if isinstance(prompts, list) else 1
        gly_lines=[]
        for i in range(batch_size): # for each batch
            texts = fonts[i]
            gly_line =[]
            for t in texts:
                gl = draw_glyph(self.fonts, t) #居中绘制
                gl = 1 - transforms.ToTensor()(gl)
                gl = gl.to(memory_format=torch.contiguous_format).float()
                gl = gl.to(device)
                gly_line += [gl] 
            gly_lines.append(gly_line) 

        # 提取要绘制的字符特征
        if self.use_gly_line:
            self.encode_caption_glyph(gly_lines)
            placeholder_token = self.tokenizer('*')
            placeholder_token = placeholder_token[0,1] # 删除[CLS]和[SEP]
            replace_clip_embeddings_mul(self.text_encoder, self.text_embs_all, placeholder_token)
                   
        text_input_ids = self.tokenizer(prompts).to(device)
        _,text_embeddings = self.text_encoder.encode_text(text_input_ids)
        add_text_embeds,text_embeddings_2048 = self.proj_zh(text_embeddings.half())
        
        # duplicate text embeddings for each generation per prompts, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompts) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompts`, but got {type(negative_prompt)} !="
                    f" {type(prompts)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompts`:"
                    f" {prompts} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompts`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input_ids = self.tokenizer(uncond_tokens).to(device)
            _,uncond_embeddings = self.text_encoder.encode_text(uncond_input_ids)
            add_text_embeds_uncond,uncond_embeddings_2048 = self.proj_zh(uncond_embeddings.half())

            # duplicate unconditional embeddings for each generation per prompts, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            text_embeddings_2048 = torch.cat([uncond_embeddings_2048, text_embeddings_2048])
            add_text_embeds = torch.cat([add_text_embeds_uncond, add_text_embeds])
         
        return text_embeddings_2048,add_text_embeds
    

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    # 输入参数增加 new_fonts_lists
    @torch.no_grad()
    def __call__(
        self,
        prompts: Union[str, List[str]],
        data_dir: str= None,
        batch_files:[List[str]]= None,
        new_fonts_lists: List[List[List[str]]] = None,
        rect_list: List[List[List[Tuple]]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0,
        image_guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        # 0. Default height and width to unet
        height = height or self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
        width = width or self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 提取font和 prompt
        pattern = r'“([^”]*)”'
        fonts = []
        replace_prompts = []
        for t in prompts:    
            font = re.findall(pattern, t)
            replace_prompt = re.sub(pattern, '*', t)
            fonts.append(font)
            replace_prompts.append(replace_prompt)

        # font = []
        # prompt = []
        # for t in prompts:
        #     font_sin = self.pattern.findall(t)
        #     if font_sin:
        #         font_sin = font_sin[0]
        #     else:
        #         font_sin = ""
        #     font.append(font_sin)
        #     prompt.append(t.replace(font_sin,"").replace("“","").replace("”",""))

        # 2. Define call parameters
        batch_size = 1 if isinstance(replace_prompts, str) else len(replace_prompts)
        device = self.pipe._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds,add_text_embeds = self.encode_prompt(replace_prompts,fonts, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt)
        prompt_embeds = prompt_embeds.half()
        add_text_embeds = add_text_embeds.half()

        # 4. Prepare timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.pipe.unet.in_channels
        latents = self.pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
            
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)

        add_time_ids = self._get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype)
        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        if controlnet_condition:
            encoder_hidden_fonts = []
            for i in range(batch_size):
                text_emb = torch.cat(self.text_embs_all[i], dim=0)    
                text_emb = torch.cat([torch.zeros(10-len(text_emb),1024).to(latents.device),text_emb]).half()
                encoder_hidden_fonts.append(text_emb)
            encoder_hidden_fonts = torch.stack(encoder_hidden_fonts)
            prompt_embeds_con = torch.cat([encoder_hidden_fonts,encoder_hidden_fonts],dim=-1)
            uncond_image_latents = torch.zeros_like(prompt_embeds_con).to(device)
            prompt_embeds_con = torch.cat([uncond_image_latents,prompt_embeds_con]) if do_classifier_free_guidance else prompt_embeds_con
        else:
            prompt_embeds_con = prompt_embeds

        if task == "glyphdraw2":
            guided_hint = self.get_guided_hint2(new_fonts_lists, rect_list, width, height, device) # 自适应框get guided hint

        else:
            guided_hint = self.load_guided_hint(data_dir, batch_files, device)

        
        for i, t in enumerate(self.pipe.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            guided_hint_input = torch.cat([torch.zeros_like(guided_hint), guided_hint]) if do_classifier_free_guidance else guided_hint
            guided_hint_input = guided_hint_input.half()

            #controlnet conditioning
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds_con,
                controlnet_cond=guided_hint_input,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            ) 

            encoder_hidden_states_all = {}
            encoder_hidden_states_all["encoder_hidden_states"] = prompt_embeds
            encoder_hidden_states_all["controlnet_attn2"] = self.controlnet_attn2

            # predict the noise residual
            # predict the noise residual
            noise_pred = self.ip_adapter(
                latent_model_input,
                t,
                encoder_hidden_states_all=encoder_hidden_states_all,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=[
                sample for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample,
                glyph_embeds=self.text_embs_all,
            )

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            # latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = self.vae.decoder.mid_block.attentions[0].processor in [
            AttnProcessor2_0,
            XFormersAttnProcessor,
            LoRAXFormersAttnProcessor,
            LoRAAttnProcessor2_0,
        ]
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if not use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(latents.dtype)
            self.vae.decoder.conv_in.to(latents.dtype)
            self.vae.decoder.mid_block.to(latents.dtype)
        else:
            latents = latents.float()

        # 8. Post-processing
        # image = self.pipe.decode_latents(latents_copy)
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="np")

        # # 9. Run safety checker
        # image, has_nsfw_concept = self.pipe.run_safety_checker(image, device, text_embeddings.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.pipe.numpy_to_pil(image)

        return image,guided_hint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="glyphdraw2",help="Glyphdraw2 evaluation set or Anytext evaluation set")
    parser.add_argument("--model_path", type=str,  default="/models/stable-diffusion-xl-protovisionXLv6",help="Base model path, download on your own")
    parser.add_argument("--device", type=str,  default="cuda")
    parser.add_argument("--selected_font", type=str,  default='./fonts/OPPOSans-S-M-0802.ttf' )
    parser.add_argument("--controlnet_condition", type=bool,  default=True)
    parser.add_argument("--isLLM", type=bool,  default=False)
    parser.add_argument("--text_encoder_path", type=str,  default="laion5B-s13B-b90k",help="Multilingual CLIP, download on your own")
    parser.add_argument("--proj_path", type=str,  default="",required=True, help="Adapters for PEA Diffusion adaptation")
    parser.add_argument("--text_proj_path", type=str,  default="", required=True, help="Weights saved for training models")
    parser.add_argument("--ctrlnet_path", type=str,  default="",required=True, help="Weights saved for training models")
    parser.add_argument("--unet_path", type=str,  default="",required=True, help="Weights saved for training models")
    parser.add_argument("--llm_model_id", type=str,  default="",help="Refer to LLaMA Factory")

    args = parser.parse_args()
    task = args.task
    selected_font = args.selected_font
    controlnet_condition = args.controlnet_condition
    isLLM = args.isLLM
    model_id = args.model_path
    text_encoder_path = args.text_encoder_path
    proj_path = args.proj_path
    text_proj_path = args.text_proj_path
    ctrlnet_path = args.ctrlnet_path
    unet_path = args.unet_path
    llm_model_id = args.llm_model_id

    sdt = StableDiffusionTest(model_id, text_encoder_path, proj_path, text_proj_path, ctrlnet_path,unet_path, args.device, selected_font)

    if task == "glyphdraw2":
        height = 1152
        width = 832
        rect_lists=[]
        negative_prompt=""
        batch_size = 2
        file_caption = "./prompts/poster_zh.txt"
        outputs = f"outputs/poster_zh"
        raw_texts = [line.strip() for line in open(file_caption).readlines()]
        os.makedirs(outputs, exist_ok=True)

        if isLLM: # LLM推断
            tokenizer = AutoTokenizer.from_pretrained(
                llm_model_id,
                revision="v2.0",
                use_fast=False,
                trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                llm_model_id,
                revision="v2.0",
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)
            model.generation_config = GenerationConfig.from_pretrained(llm_model_id, revision="v2.0")
        
        pattern = r'“([^”]*)”'
        for i in range(0, len(raw_texts)-1, batch_size):
            text_batch = raw_texts[i:(i + batch_size)]
            if isLLM: # LLM infers rectangular boxes
                rect_lists, new_fonts_lists = inferenceLLMbbox(text_batch=text_batch, w=width, h=height, tokenizer=tokenizer, model=model, isnorm=False)
            else: # Adaptive generation of rectangular boxes
                rect_lists = []
                new_fonts_lists = []
                for i in range(batch_size):
                    prompt = text_batch[i]
                    string_list =  re.findall(pattern, prompt)
                    rectangles, new_fonts = adaptive_generate_rectangles2(1024, 1024, string_list,selected_font)
                    new_fonts_lists.append(new_fonts)
                    rects = []
                    for r in rectangles:
                        rects.append(get_rectangle_vertices(r))
                    rect_lists.append(rects) 
            
            images,guided_hints = sdt(text_batch, new_fonts_lists=new_fonts_lists, rect_list=rect_lists, height=height, width=width, negative_prompt=[negative_prompt]*len(text_batch))
            for i,(image,guided_hint) in enumerate(zip(images,guided_hints)):
                if i>len(images)-1:continue
                name = text_batch[i].strip()[:80].replace("/","")
                image.save(f"{outputs}/{name}.jpg", normalize=True)
                save_image(guided_hint,f"{outputs}/{name}.png")
    else:
        batch_size = 5
        height = 1024
        width = 1024
        rect_lists=[]
        negative_prompt=""
        en = True
        if en:
            file_caption = "/mnt/data/group/yonglin/glyphdraw_test/output_laion.txt"
            image_files = get_image_paths('/mnt/data/group/yonglin/glyphdraw_test/benchmark/laion_word/test1k.json')
            data_dir = '/mnt/data/group/yonglin/glyphdraw_test/benchmark/laion_word/glyph_laion_1024'
            outputs = f"outputs/anytext_en_52"
            raw_texts = [line.strip() for line in open(file_caption).readlines()]
        else:
            file_caption = "/mnt/data/group/yonglin/glyphdraw_test/output_wukong.txt"
            image_files = get_image_paths("/mnt/data/group/yonglin/glyphdraw_test/benchmark/wukong_word/filtered_data.json")
            data_dir = "/mnt/data/group/yonglin/glyphdraw_test/benchmark/wukong_word/wukong_laion_1024_2"
            outputs = f"outputs/table1/anytext_all_52"
            raw_texts = [line.strip() for line in open(file_caption).readlines()]
        os.makedirs(outputs, exist_ok=True)

        for i in range(0, len(raw_texts)-1, batch_size):
            text_batch = raw_texts[i:(i + batch_size)] # prompts
            batch_files = image_files[i:i+batch_size] # guided hint images
            images,guided_hints = sdt(text_batch, data_dir=data_dir, batch_files=batch_files, height=height, width=width, negative_prompt=[negative_prompt]*len(text_batch))
            for i,(image,guided_hint) in enumerate(zip(images,guided_hints)):
                name = text_batch[i].strip()[:80].replace("/","")
                image.save(f"{outputs}/{name}.jpg", normalize=True)
                save_image(guided_hint,f"{outputs}/{name}.png")
