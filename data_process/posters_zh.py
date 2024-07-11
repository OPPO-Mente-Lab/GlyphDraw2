# Copyright (c) OPPO Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os,re,glob
import paddle
paddle.utils.run_check()

from randeng.modeling_deltalm import DeltalmForConditionalGeneration
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BlipImageProcessor

import torch
from torchvision import transforms as T
from torchdata.datapipes.iter import FileOpener
from torchdata.dataloader2 import MultiProcessingReadingService, DataLoader2
import webdataset
from PIL import Image
import os,re
import tqdm
import argparse
import timm
import torch.nn as nn
import torch.nn.functional as F
import json
import zhconv
import time

from paddleocr import PaddleOCR, draw_ocr
from paddleocr.paddleocr import get_model_config, parse_args
from paddleocr.tools.infer.predict_rec import TextRecognizer
from paddleocr.tools.infer.utility import get_rotate_crop_image, get_minarea_rect_crop

import cv2
import copy
from shapely.geometry import Polygon
from timm.models.efficientnet import _cfg

from omegaconf import OmegaConf
import numpy as np
import yaml
from torchvision.utils import save_image
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint


def decode(item):
    key, value = item

    if key.endswith(".jpg"):
        try:
            value = Image.open(value).convert("RGB")
        except Exception as e:
            print(f"Reading {key} error, skip.")
            value = None
        return key, value
    else:
        value = None
        return key, value

def filter_resolution(example):
    if example is None:
        return False
    if example.size[0] < 704 or example.size[1] < 704:
        return False
    if example.size[0] * example.size[1] < 1024*1024:
        return False
    return True

def has_chinese_char(s):
    for c in s:
        if '\u4e00' <= c <= '\u9fff':
            return True
    return False

def load_watermark_model(model_path='models/watermark_model_v1.pt'):
    config = _cfg(url='', file="model/efficientnet_b3_ra2-cf984f9c.pth")
    model = timm.create_model('efficientnet_b3a', pretrained=True,pretrained_cfg=config, num_classes=2)
    # model = timm.create_model('efficientnet_b3a', pretrained=True, num_classes=2)

    model.classifier = nn.Sequential(
        # 1536 is the orginal in_features
        nn.Linear(in_features=1536, out_features=625),
        nn.ReLU(),  # ReLu to be the activation function
        nn.Dropout(p=0.3),
        nn.Linear(in_features=625, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=2),
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    preprocess = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return model, preprocess

def load_image_mask(img,box_no):
    # from matplotlib.patches import Polygon
    img_np = np.array(img.convert('RGB'))
    if img_np.ndim == 3:
        img_np = np.transpose(img_np, (2, 0, 1))
    out_img = img_np.astype('float32') / 255

    img_width, img_height = img.size
    mask_img = np.zeros((img_height, img_width), dtype=np.uint8)
    rgb = (255, 255, 255)
    for box in box_no:
        box = box.astype(np.int32)
        cv2.fillPoly(mask_img, [box], (255))
    # mask_img = Image.fromarray(mask_img.astype(np.uint8))
    mask_img = mask_img[None,:].astype('float32') / 255

    out_img = pad_img_to_modulo(out_img, 8)
    mask_img = pad_img_to_modulo(mask_img, 8)

    batch={}
    batch["image"]  = torch.tensor(out_img).unsqueeze(0)
    batch["mask"] = torch.tensor(mask_img).unsqueeze(0)
    return batch

def lama_model(batch):
    with torch.no_grad():
        batch = move_to_device(batch, device)
        batch = model(batch)                    
        cur_res = batch["inpainted"][0].permute(1, 2, 0).detach().cpu().numpy()
        unpad_to_size = batch.get('unpad_to_size', None)
        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        # cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cur_res = Image.fromarray(cur_res.astype(np.uint8))
        # cv2.imwrite("1.png", cur_res)
    return cur_res

def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')

def contains_alpha(string):
    pattern = '[a-zA-Z]'
    match = re.search(pattern, string)
    return bool(match)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Data Processing Pipeline", add_help=True)

    parser.add_argument('--process_id', default=0, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--num_processors', default=8, type=int)
    parser.add_argument('--data_path', default="/mnt/data/group/text2img_data/data_font/haibao_ori/2_tusij_50W", type=str) 
    parser.add_argument('--output_path', default="/mnt/data/group/text2img_data/data_font/2_tusij_process", type=str) 

    args = parser.parse_args()
    process_id = args.process_id
    batch_size = args.batch_size
    num_processors = args.num_processors
    data_path = args.data_path
    output_path = args.output_path
    device = "cuda"

    # load model
    blip_model = Blip2ForConditionalGeneration.from_pretrained("/mnt/data/group/models/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)
    processor = Blip2Processor.from_pretrained("/mnt/data/group/models/blip2-opt-2.7b")
    image_processor = BlipImageProcessor.from_pretrained("/mnt/data/group/models/blip2-opt-2.7b")

    translation_model = DeltalmForConditionalGeneration.from_pretrained("/mnt/data/group/models/Randeng-Deltalm-362M-En-Zh", torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained("/mnt/data/group/models/infoxlm-base")

    ocr = PaddleOCR(use_angle_cls=True, lang="ch", det_model_dir="/mnt/data/group/majian/glyphdraw2/model_ocr/ch_PP-OCRv4_det_infer", \
                    rec_model_dir="/mnt/data/group/majian/glyphdraw2/model_ocr/ch_PP-OCRv4_rec_infer",  \
                    cls_model_dir="/mnt/data/group/majian/glyphdraw2/model_ocr/ch_ppocr_mobile_v2.0_cls_infer", show_log=False)  # need to run only once to download and load model into memory

    # watermark_detector, watermark_process = load_watermark_model("/data_share/liangjunhao/image_filter/watermark_model_v1.pt")
    # watermark_detector = watermark_detector.to(device, dtype=torch.float16)

    model_path = "/mnt/data/group/models/big-lama/big-lama"
    train_config_path = os.path.join(model_path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'
    checkpoint_path = os.path.join(model_path, 'models', 'best.ckpt')
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)

    def collate_fn(examples):

        # txt = [example[".txt"] if ".txt" in example else "" for example in examples ]
        key = [example["__key__"] for example in examples]
        jpg = [example[list(example.keys())[1]] for example in examples ]

        return { "jpg": jpg, "key": key}

    output_path = f"{output_path}/{process_id+8}"
    os.makedirs(output_path, exist_ok=True)

    tar_list = sorted([os.path.join(data_path, tar_name) for tar_name in os.listdir(data_path) if tar_name.endswith("gz")])
    curr_tar_list = np.array_split(tar_list, num_processors)[process_id][3:]
    rs = MultiProcessingReadingService(num_workers=1)

    index = 0
    num_per_dir = 10000
    pattern = os.path.join(output_path, f"%05d.tar")
    sink = webdataset.ShardWriter(pattern, maxsize=int(6e9), maxcount=int(num_per_dir))

    for tar_index,tar_name in enumerate(curr_tar_list):
        dataset = FileOpener([tar_name], mode="b").load_from_tar().map(decode).webdataset().batch(batch_size).collate(collate_fn=collate_fn)
        
        dl = DataLoader2(dataset, reading_service=rs)
        names = os.path.basename(tar_name).replace(".gz","")
        # sink = webdataset.TarWriter(os.path.join(output_path,names))
        # captions_path = os.path.join(output_path, os.path.basename(tar_name)).replace(f"{data}",f"{data}_txt").replace("tar","txt")
        # captions = open(captions_path,"w",encoding="utf-8")
        print(f"started {tar_name}")
        for obj in tqdm.tqdm(dl):
            with torch.no_grad():
                jpgs = obj["jpg"]
                # txts = obj["__key__"].split("_")[0]
                # txts = obj["txt"]
                pixel_values = []
                jpgs_new = []
                txts_new = []
                font_infors = []
                fonts = []
                for j,img in enumerate(jpgs):
                    img_np = np.array(img)
                    try:
                        img_h, img_w = img_np.shape[0], img_np.shape[1]
                        dt_boxes, rec_res, _ = ocr(img_np, cls=False)
                    except:
                        print("error",img_np)
                        continue
                    if len(rec_res)==0 or len(rec_res)>20:continue 
                    qualified_res = []
                    chars = []
                    box_no = []
                    text_no = []
                    for box, rec1 in zip(dt_boxes, rec_res):
                        corners_text = " ".join([f"({int(x):d},{int(y):d})" for x, y in box])
                        ocr_text, conf_score = rec1
                        center = np.mean(box, axis=0)
                        area = Polygon(box).area/(img_h*img_w)
                        if has_chinese_char(ocr_text) or contains_alpha(ocr_text):
                            char_areas = area / len(re.sub('[^\w\s]', '', ocr_text))
                        else:
                            char_areas = 0          
                        if "@" in ocr_text or "有限" in ocr_text or "com" in ocr_text:
                            box_no.append(box)
                            text_no.append(ocr_text)
                            continue
                        if len(ocr_text) <= 40 and char_areas > 0.001:
                            qualified_res.append((ocr_text, corners_text))
                            chars.append(ocr_text)
                        else:
                            box_no.append(box)
                            text_no.append((ocr_text,char_areas))
                    if box_no:
                        batch = load_image_mask(img,box_no) # img.size=w,h  batch["image"].shape=1*3*h*w
                        try:
                            img = lama_model(batch)
                        except:
                            print("lama error")
                            continue

                    if len(qualified_res)>0:
                        # watermark_image = watermark_process(img).unsqueeze(0)
                        # watermark_score = F.softmax(watermark_detector(watermark_image.to(device, dtype=torch.float16)), dim=1)
                        # if watermark_score[:, 0]<0.3:
                        jpgs_new.append(img)
                        # txts_new.append(txts[j])
                        font_infors.append(qualified_res)
                        fonts.append(chars)
                      
                if len(jpgs_new)==0:continue
                # BLIP2 Caption
                pixel_values = image_processor(jpgs_new, return_tensors="pt").to(device, torch.float16).pixel_values
                generated_ids = blip_model.generate(pixel_values=pixel_values)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, max_new_tokens=32)
                # BLIP2 Caption to Chinese
                translation_inputs = tokenizer(generated_text, max_length=64, truncation=True, padding=True, return_tensors="pt")
                generate_ids = translation_model.generate(translation_inputs["input_ids"].to(device), max_new_tokens=64)
                caption_zhs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for i in range(len(jpgs_new)):
                key ="%08d" % index
                sink.write({
                    "__key__": key,
                    "txt": caption_zhs[i],
                    "jpg": jpgs_new[i],
                    "text": " && ".join(fonts[i]),
                    "json": {"caption_en": generated_text[i], "caption_zh": caption_zhs[i], 
                    "caption_ori_zh": "", "font": font_infors[i]}
                })
                index += 1
        print(f"finished {tar_name}")
    dl.shutdown()
    sink.close()
