
# Copyright (c) OPPO Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from third_party.custom_multiplexer import SampleMultiplexer
import braceexpand
import webdataset as wds
from tqdm import tqdm
import torch
from torchvision.transforms.functional import crop
import re
from torchvision import transforms
import random
import zhconv
import numpy as np
import os,glob
from transformers import T5Tokenizer, T5ForConditionalGeneration,T5EncoderModel,MT5EncoderModel,AutoTokenizer,AutoModel,AutoModelForCausalLM
from torchdata.datapipes.iter import FileLister, FileOpener
from torchdata.datapipes.iter import IterableWrapper
from torchvision.utils import save_image

from pytorch_lightning import LightningDataModule
from typing import Optional
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, DistributedReadingService, SequentialReadingService
from torch.utils.data import random_split
import cv2
from PIL import Image
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import einops
import torchvision.transforms.functional as TF

USED_KEYS = ["txt","jpg","text","json"]

# BUCKETS = [[576, 1792], [640, 1600], [704, 1408], [768, 1280], [832, 1152], [896, 1088], [960, 1024], [1024, 1024], [1024, 960], [1088, 896], [1152, 832], [1280, 768], [1408, 704], [1600, 640], [1792, 576]]
# BUCKET_PROBS = [0.001017367923842744, 0.0006540222367560497, 0.003052103771528232, 0.018603299178838746, 0.08843834023690139, 0.019475328827846812, 0.00944698786425405, 0.24369595232904587, 0.012353753360947605, 0.028558971005014172, 0.38478308262480926, 0.15126080953419083, 0.030230361165612965, 0.006140542111765133, 0.002289077828646174]
# Poster data resolution distribution statistics
BUCKETS = [[704, 1408], [832, 1152], [1024, 1024], [1152, 832]]
BUCKET_PROBS = [0.165, 0.442, 0.208, 0.185]
MAX_AR_ERROR = 2
ASPECTS = np.array([b[0]/b[1] for b in BUCKETS])
MAX_lines = 15
phrase_list = [
    ', 文本的内容是',
    ', 图像中描绘的文本材料是',
    ', 文本说',
    ', 快照中显示的标题是',
    ', 有这些话：',
    ', 上面写着',
    ', 图片上的文本材料：',
    ', 上面写着这些文本：',
    ', 标题是',
    ', 图中的文本内容为'
]

def str_contain_chinese(str):
    for ch in str:
        if u'\u4e00'<=ch<=u'\u9fff':
            return True
    return False

def get_consume_samples(data_model: LightningDataModule) -> int:
    if hasattr(data_model.trainer.lightning_module, 'consumed_samples'):
        consumed_samples = data_model.trainer.lightning_module.consumed_samples
        print('get consumed samples from model: {}'.format(consumed_samples))
    else:
        world_size = data_model.trainer.world_size
        consumed_samples = max(0, data_model.trainer.global_step - 1) * \
            data_model.hparams.train_batchsize * world_size * \
            data_model.trainer.accumulate_grad_batches
        print('calculate consumed samples: {}'.format(consumed_samples))
    return consumed_samples

def split_bucket(x):
    return x["bucket_id"]


class DataModuleCustom(LightningDataModule):
    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Universal DataModule')
        parser.add_argument('--webdataset_base_urls', type=str, nargs="+")
        parser.add_argument('--num_workers', default=1, type=int)
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--shard_width', default=5, type=int)
        parser.add_argument('--hr_size', default=-1, type=int)
        parser.add_argument('--train_split', default=1.0, type=float)
        parser.add_argument('--val_split', default=0.0, type=float)
        parser.add_argument('--test_split', default=0.0, type=float)
        parser.add_argument('--shuffle_train',default=False, action="store_true")
        parser.add_argument('--resample_train',default=False, action="store_true")
        parser.add_argument('--shuffle_num', default=None, type=int)
        parser.add_argument('--test_prompts', type=str,
                            default="./test_prompts.txt")
        parser.add_argument('--test_repeat', default=1, type=int)

        parser.add_argument(
            "--resolution", type=int, default=512,
            help=(
                "The resolution for input images, all the images in the train/validation dataset will be resized to this"
                " resolution"
            ),
        )
        parser.add_argument(
            "--center_crop", action="store_true", default=False,
            help="Whether to center crop images before resizing to resolution"
        )
        return parent_args

    def __init__(
        self,
        args,
        tokenizer,
        custom_collate_fn=None,
    ):
        super().__init__()
        # self.available_shards = list(range(args.start_shard, args.end_shard + 1))
        # if splits is None:
        #     splits = []
        splits = {
            'train': args.train_split,
            'val': args.val_split,
            'test': args.test_split,
        }
        self.webdataset_base_urls = args.webdataset_base_urls
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.shuffle_train = args.shuffle_train
        self.resample_train = args.resample_train
        self.shard_width = args.shard_width
        self.hr_size = args.hr_size
        self.shuffle_num = args.shuffle_num
        self.tokenizer = tokenizer
        self.collate_fn = custom_collate_fn if custom_collate_fn is not None else collate_fn
        self.center_crop = args.center_crop
        self.resolution = args.resolution

        self.train_prop = self.val_prop = self.test_prop = 0
        self.datasets = {}
        if splits['train'] > 0:
            self.train_prop = splits['train']
            self.train_dataloader = self._train_dataloader
            self.datasets['train'] = None


        self.prepare_data()
        self.setup()

    def prepare_data(self):
        assert self.train_prop + self.test_prop + self.val_prop == 1

        all_urls = []
        for url in self.webdataset_base_urls:
            if "*" in url:
                all_urls += expand_urls1(url)
            else:
                all_urls += expand_urls(url)
        num_train = round(self.train_prop*len(all_urls))
        num_test = round(self.test_prop*len(all_urls))
        num_val = len(all_urls) - num_train - num_test
        assert num_train + num_test + \
            num_val == len(
                all_urls), f"{num_train} + {num_test} + {num_val} = {num_train + num_test + num_val} != {len(all_urls)}"
        self.train_urls, self.test_urls, self.val_urls = random_split(
            all_urls, [num_train, num_test, num_val])  # , generator=torch.Generator().manual_seed(self.seed)

    def setup(self, stage=None):
        if 'train' in self.datasets:
            self.datasets['train'] = ImageEmbeddingDataset(
                self.train_urls,
                self.tokenizer,
                shuffle_shards=self.shuffle_train,
                resample=self.resample_train,
                hr_size=self.hr_size,
                handler=wds.handlers.warn_and_continue,
                center_crop=self.center_crop,
                size=self.resolution,
            )

            if self.shuffle_num is not None and self.shuffle_num > 0:
                self.datasets['train'].shuffle(self.shuffle_num)


    def _train_dataloader(self):
        # return self.create_dataloader(self.train_urls, shuffle=self.shuffle_train, resample=self.resample_train)
        pipes_to_weights_dict = {}

        dp_list = IterableWrapper(self.datasets['train']).mydemux(
            num_instances=len(BUCKET_PROBS), classifier_fn=split_bucket, buffer_size=1000)

        for i in range(len(dp_list)):
            pipes_to_weights_dict[dp_list[i]] = BUCKET_PROBS[i]
        sample_mul_dp = SampleMultiplexer(
            pipes_to_weights_dict=pipes_to_weights_dict, batch_size=self.batch_size, seed=0).collate(collate_fn=collate_fn)
        mp_rs = MultiProcessingReadingService(num_workers=self.num_workers)
        dist_rs = DistributedReadingService()
        rs = SequentialReadingService(dist_rs, mp_rs)
        return DataLoader2(sample_mul_dp, reading_service=rs)

def expand_urls(urls):
    if isinstance(urls, str):
        urllist = urls.split("::")
        result = []
        for url in urllist:
            result.extend(braceexpand.braceexpand(url))
        return result
    else:
        return list(urls)

def expand_urls1(urls):
    result = []
    for file_ in glob.glob(urls):
        result.append(file_)
    return result

def verify_keys(samples, required_keys, handler=wds.handlers.reraise_exception):
    for sample in samples:
        sample_json = sample["json"]
        if len(sample_json["font"]) > MAX_lines:continue
        w, h = sample["jpg"].size

        is_normal = True
        aspect = float(w)/float(h)
        bucket_id = np.abs(ASPECTS - aspect).argmin()
        if abs(ASPECTS[bucket_id] - aspect) < MAX_AR_ERROR:
            sample["bucket_id"] = bucket_id
        for key in required_keys:
            if key not in sample:
                print(f"Sample {sample['__key__']} missing {key}. Has keys {sample.keys()}")
                is_normal = False
        if is_normal:
            yield {key: sample[key] for key in required_keys}

def crop_left_upper(image, size):
    w, h = image.size
    # The maximum distance that can be moved horizontally and vertically during cropping
    detla_w = w-size[0]
    detla_h = h-size[1]
    # Crop starting coordinates
    x = random.randint(0, detla_w)
    y = random.randint(0, detla_h)
    return (y, x), crop(image, y, x, size[1], size[0])


def get_caption(ori_caption, num, place_holder='*'):
    new_caption = ori_caption + random.choice(phrase_list)
    pos=''
    for i in range(num):
        pos += place_holder + ' , '
    pos = pos[:-2] + '.'
    new_caption += pos
    return new_caption

def adaptive_crop(image, polygons, fonts, size, ratio=1.0):
    numbers = [len(f) for f in fonts]
    max_number = max(numbers)
    max_polygon = None
    for polygon, number in zip(polygons, numbers):
        if number == max_number:
            max_polygon = polygon
            break

    if max_polygon is None:
        return None 

    rect = cv2.boundingRect(np.array(max_polygon))
    x, y, w, h = rect
    center_x = x + w // 2
    center_y = y + h // 2
    crop_width = int(size[0] * ratio)
    crop_height = int(size[1] * ratio)
    crop_x = max(center_x - crop_width // 2, 0)
    crop_y = max(center_y - crop_height // 2, 0)
    if crop_x + crop_width > image.size[0]:
        crop_x = max(image.size[0] - crop_width, 0)
    if crop_y + crop_height > image.size[1]:
        crop_y = max(image.size[1] - crop_height, 0)

    cropped_image = crop(image, crop_y, crop_x, crop_height, crop_width)
    resized_image = transforms.Resize((size[1], size[0]), interpolation=transforms.InterpolationMode.BILINEAR)(cropped_image)

    return (crop_y, crop_x),resized_image

def crop_mask(image, start_points, size, ratio=1.0):
    crop_width = int(size[0] * ratio)
    crop_height = int(size[1] * ratio)

    cropped_image = crop(image, start_points[0], start_points[1], crop_height, crop_width)
    resized_image = transforms.Resize((size[1], size[0]), interpolation=transforms.InterpolationMode.BILINEAR)(cropped_image)
    return resized_image

# Draw characters centered
def draw_glyph(font, text):
    g_size = 50
    W, H = (512, 80)
    new_font = font.font_variant(size=g_size)
    img = Image.new(mode='1', size=(W, H), color=0) 
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = new_font.getbbox(text)
    text_width = max(right-left, 5)
    text_height = max(bottom - top, 5)
    ratio = min(W*0.9/text_width, H*0.9/text_height)
    new_font = font.font_variant(size=int(g_size*ratio))

    text_width, text_height = new_font.getsize(text)
    offset_x, offset_y = new_font.getoffset(text)
    x = (img.width - text_width) // 2
    y = (img.height - text_height) // 2 - offset_y//2
    draw.text((x, y), text, font=new_font, fill='white')
    # img.save(f'./test_images/gly_lines/{text}.jpg')
    img = np.expand_dims(np.array(img), axis=2).astype(np.float64)
    # cv2.imwrite(f'./test_images/gly_line/{text}.jpg', img)
    return img

def insert_spaces(string, nSpace):
    if nSpace == 0:
        return string
    new_string = ""
    for char in string:
        new_string += char + " " * nSpace
    return new_string[:-nSpace]

def draw_glyph2(font, text, polygon, vertAng=10, scale=1,ratio=1, width=512, height=512, add_space=True):
    rect = cv2.minAreaRect(np.array(polygon))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    w, h = rect[1]
    angle = rect[2]
    if angle < -45:
        angle += 90
    angle = -angle
    if w < h:
        angle += 90

    vert = False
    if (abs(angle) % 90 < vertAng or abs(90-abs(angle) % 90) % 90 < vertAng):
        _w = max(box[:, 0]) - min(box[:, 0])
        _h = max(box[:, 1]) - min(box[:, 1])
        if _h >= _w:
            vert = True
            angle = 0

    img = np.zeros((height, width, 3), np.uint8)
    # img = np.zeros((height*scale, width*scale, 3), np.uint8)
    img = Image.fromarray(img)

    # infer font size
    image4ratio = Image.new("RGB", img.size, "white")
    draw = ImageDraw.Draw(image4ratio)
    _, _, _tw, _th = draw.textbbox(xy=(0, 0), text=text, font=font)
    text_w = min(w, h) * (_tw / _th)
    if text_w <= max(w, h):
        # add space
        if len(text) > 1 and not vert and add_space:
            for i in range(1, 100):
                text_space = insert_spaces(text, i)
                _, _, _tw2, _th2 = draw.textbbox(xy=(0, 0), text=text_space, font=font)
                if min(w, h) * (_tw2 / _th2) > max(w, h):
                    break
            text = insert_spaces(text, i-1)
        font_size = min(w, h)*0.8  ## 0.8
    else:
        shrink = 0.75 if vert else 0.85  ## 0.75 0.85
        font_size = min(w, h) / (text_w/max(w, h)) * shrink
    new_font = font.font_variant(size=int(font_size))

    left, top, right, bottom = new_font.getbbox(text)
    text_width = right-left
    text_height = bottom - top

    layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if not vert:
        draw.text((rect[0][0]-text_width//2, rect[0][1]-text_height//2-top), text, font=new_font, fill=(255, 255, 255, 255))
    else:
        x_s = min(box[:, 0]) + _w//2 - text_height//2
        y_s = min(box[:, 1])
        for c in text:
            draw.text((x_s, y_s), c, font=new_font, fill=(255, 255, 255, 255))
            _, _t, _, _b = new_font.getbbox(c)
            y_s += _b

    rotated_layer = layer.rotate(angle, expand=1, center=(rect[0][0], rect[0][1]))

    x_offset = int((img.width - rotated_layer.width) / 2)
    y_offset = int((img.height - rotated_layer.height) / 2)
    img.paste(rotated_layer, (x_offset, y_offset), rotated_layer)    
    img = np.expand_dims(np.array(img.convert('1')), axis=2).astype(np.float64)
    img = img*255

    return img

key_verifier = wds.filters.pipelinefilter(verify_keys)

class ImageEmbeddingDataset(wds.DataPipeline, wds.compat.FluidInterface):
    """
    A fluid interface wrapper for DataPipline that retontsurns image embedding pairs
    Reads embeddings as npy files from the webdataset if they exist. If embedding_folder_url is set, they will be inserted in from the alternate source.
    """

    def __init__(
            self,
            urls,
            tokenizer=None,
            extra_keys=["bucket_id"],
            hr_size=-1,
            size=512,
            handler=wds.handlers.reraise_exception,
            resample=False,
            shuffle_shards=True,
            center_crop=False,
            font_path = './fonts/OPPOSans-S-M-0802.ttf',
            glyph_scale=0.08,
            ratio=0.08,
            max_chars=15
    ):

        super().__init__()
        keys = USED_KEYS + extra_keys
        # self.key_map = {key: i for i, key in enumerate(keys)}
        self.font = ImageFont.truetype(font_path, size=60)
        self.glyph_scale = glyph_scale
        self.ratio = ratio
        self.max_lines = MAX_lines
        self.resampling = resample
        self.hr_size = hr_size
        self.center_crop = center_crop
        self.crop = transforms.CenterCrop(size) if center_crop else crop_left_upper
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.image_transforms_mask_nocrop = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.functional._interpolation_modes_from_int(0)),
                # transforms.CenterCrop(size)
            ]
        )

        self.image_transforms_mask = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.functional._interpolation_modes_from_int(0)),
                transforms.CenterCrop(size)
            ]
        )

        self.aspects = np.array([b[0]/b[1] for b in BUCKETS])
        self.tokenizer = tokenizer

        if resample:
            assert not shuffle_shards, "Cannot both resample and shuffle"
            self.append(wds.ResampledShards(urls))
        else:
            self.append(wds.SimpleShardList(urls))
            if shuffle_shards:
                self.append(wds.filters.shuffle(1000))

        self.append(wds.tarfile_to_samples(handler=handler))

        self.append(wds.decode("pilrgb", handler=handler))

        self.append(key_verifier(required_keys=keys, handler=handler))
        # Apply preprocessing
        self.append(wds.map(self.preproc))
        # self.append(wds.to_tuple(*keys))

    def draw_inv_mask(self, polygons):
        img = np.zeros((512, 512))
        for p in polygons:
            pts = p.reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], color=255)
        img = img[..., None]
        return img/255.

    def draw_pos(self, ploygon, text, prob=1.0, width=512, height=512):
        img = np.zeros((height, width))
        
        rect = cv2.minAreaRect(np.array(ploygon))
        w, h = rect[1]
        small = False
        if w < 20 or h < 20:
            small = True
        if random.random() < prob:
            pts = np.array(ploygon).reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], color=255)
            # 10% dilate / 10% erode / 5% dilatex2  5% erodex2
            random_value = random.random()
            kernel = np.ones((3, 3), dtype=np.uint8)
            if random_value < 0.7:
                pass
            elif random_value < 0.8:
                img = cv2.dilate(img.astype(np.uint8), kernel, iterations=1)
            elif random_value < 0.9 and not small:
                img = cv2.erode(img.astype(np.uint8), kernel, iterations=1)
            elif random_value < 0.95:
                img = cv2.dilate(img.astype(np.uint8), kernel, iterations=2)
            elif random_value < 1.0 and not small:
                img = cv2.erode(img.astype(np.uint8), kernel, iterations=2)
        img = img[..., None] 

        return img

    def get_hint(self, positions):
        if len(positions) == 0:
            return np.zeros((512, 512, 1))
        return np.sum(positions, axis=0).clip(0, 1)


    def preproc(self, sample):
        """Applies the preprocessing for images"""

        example = {}
        instance_image = sample["jpg"]
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["original_size"] = instance_image.size
        example["bucket_id"] = sample["bucket_id"]
        
        w, h = example["original_size"][0], example["original_size"][1]

        parsed_data = sample["json"]
        font_data = parsed_data["font"]
        polygons = [
            [tuple(map(int, re.findall(r'\d+', coordinate))) for coordinate in item[1].split()]
            for item in font_data
        ]
        # Extract rendered characters
        text_list = []
        for item in font_data:
            text = item[0]
            text_list.append(text)
        example["instance_prompt_glyph"] = " && ".join(text_list)

        # example['glyphs']=[]
        example['gly_line']=[]
        # example['positions']=[]
        lps=[]
        lgs=[]
        # glyphs
        for idx, text in enumerate(text_list):
            gly_line = draw_glyph(self.font, text)
            glyphs = draw_glyph2(self.font, text, polygons[idx], scale=self.glyph_scale,ratio = self.ratio, width=w, height=h)
            lgs += [glyphs]
            # example['glyphs'] += [glyphs]
            example['gly_line'] += [gly_line]
        # mask_pos
        for idx, polygon in enumerate(polygons):
            lp = self.draw_pos(polygon, text=text_list[idx], prob=1.0 ,width=w,height=h)
            lps += [lp]
            # example['positions'] += [lp]
                
        lgs = np.array(lgs)
        lg_result = np.sum(lgs, axis=0)
        lps = np.array(lps)
        lp_result = np.sum(lps, axis=0)
        lg_img = Image.fromarray(lg_result.astype(np.uint8).squeeze(), mode='L')
        lp_img = Image.fromarray(lp_result.astype(np.uint8).squeeze(), mode='L')

        dst_size = BUCKETS[sample["bucket_id"]]
        # resize original image &  lp & lg
        if(example["original_size"][0] < dst_size[0]): #width
            if(example["original_size"][1] < dst_size[1]): #height
                instance_image = transforms.Resize((dst_size[1], dst_size[0]), interpolation=transforms.InterpolationMode.BILINEAR)(instance_image)
                lg_img = transforms.Resize((dst_size[1], dst_size[0]), interpolation=transforms.InterpolationMode.BILINEAR)(lg_img)
                lp_img = transforms.Resize((dst_size[1], dst_size[0]), interpolation=transforms.InterpolationMode.BILINEAR)(lp_img)
            else:
                instance_image = transforms.Resize((example["original_size"][1], dst_size[0]), interpolation=transforms.InterpolationMode.BILINEAR)(instance_image)
                lg_img = transforms.Resize((example["original_size"][1], dst_size[0]), interpolation=transforms.InterpolationMode.BILINEAR)(lg_img)
                lp_img = transforms.Resize((example["original_size"][1], dst_size[0]), interpolation=transforms.InterpolationMode.BILINEAR)(lp_img)
        else:
            if(example["original_size"][1] < dst_size[1]): #height
                instance_image = transforms.Resize((dst_size[1], example["original_size"][0]), interpolation=transforms.InterpolationMode.BILINEAR)(instance_image)
                lg_img = transforms.Resize((dst_size[1], example["original_size"][0]), interpolation=transforms.InterpolationMode.BILINEAR)(lg_img)
                lp_img = transforms.Resize((dst_size[1], example["original_size"][0]), interpolation=transforms.InterpolationMode.BILINEAR)(lp_img)

        ratio1 = instance_image.size[0] / dst_size[0]
        ratio2 = instance_image.size[1] / dst_size[1]
        diate_ratio = min(ratio1, ratio2)
        example["crops_coords_top_left"], instance_image = adaptive_crop(instance_image, polygons=polygons, fonts=text_list, size=dst_size, ratio=diate_ratio)
        # example["crops_coords_top_left"], instance_image = self.crop(instance_image, dst_size)

        example["instance_images"] = self.image_transforms(instance_image)

        # crop lg & lp
        crop_lg_img = crop_mask(lg_img, example["crops_coords_top_left"], dst_size, diate_ratio)
        crop_lp_img = crop_mask(lp_img, example["crops_coords_top_left"], dst_size, diate_ratio)

        example['glyphs'] = [np.expand_dims(np.array(crop_lg_img), axis=2).astype(np.float64)]
        example['positions'] = [np.expand_dims(np.array(crop_lp_img), axis=2).astype(np.float64)]

        # padding
        n_lines = min(len(text_list), self.max_lines)
        example['n_lines'] = n_lines
        all_glyphs = np.array(example['glyphs'])
        all_glyph_result = np.sum(all_glyphs, axis=0)
        # all_gly_line = np.array(example['gly_line'])
        # all_gly_line_result = np.sum(all_gly_line, axis=0)
        all_positions = np.array(example['positions'])
        all_positions_result = np.sum(all_positions, axis=0)

        if not sample["txt"] and "caption_en" in parsed_data:
            sample["txt"] = parsed_data["caption_en"]
        else:
            sample["txt"] = sample["txt"].strip()

        count=0
        if '*' in sample["txt"]:
            count += 1
            sample["txt"] = sample["txt"].replace("*", " ")
        if count > 0:
            print(f"Found {count} image's caption contain placeholder: *, change to ' '...")

        new_caption = get_caption(sample["txt"], n_lines)
        example["instance_prompt_ids"] = new_caption
        
        example["input_ids"] = self.tokenizer([example["instance_prompt_ids"]])
        example["input_ids_uncond"] = self.tokenizer([""])

        all_glyph_result = transforms.ToTensor()(all_glyph_result.astype(np.uint8))
        all_positions_result = transforms.ToTensor()(all_positions_result)

        all_glyph_result = all_glyph_result.to(memory_format=torch.contiguous_format).float()
        all_positions_result = all_positions_result.to(memory_format=torch.contiguous_format).float()

        example["glyphs"] = all_glyph_result
        # example["gly_line"] = all_gly_line_result
        example["positions"] = all_positions_result

        for i in range(len(example["gly_line"])):
            example['gly_line'][i] = 1 - transforms.ToTensor()(example['gly_line'][i])
            example['gly_line'][i] = example['gly_line'][i].to(memory_format=torch.contiguous_format).float()

        return example


def collate_fn(examples):
    instance_prompt = [example["instance_prompt_ids"] for example in examples]
    instance_prompt_glyph = [example["instance_prompt_glyph"] for example in examples]
    original_size = [example["original_size"] for example in examples]
    input_ids = [example["input_ids"] for example in examples]
    input_ids_uncond = [example["input_ids_uncond"] for example in examples]

    pixel_values = [example["instance_images"] for example in examples]
    crops_coords_top_left = [example["crops_coords_top_left"]for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    glyphs = [example["glyphs"] for example in examples]
    gly_line = [example["gly_line"] for example in examples]
    positions = [example["positions"] for example in examples]
    glyphs = torch.stack(glyphs)
    positions = torch.stack(positions)
    n_lines = [example["n_lines"] for example in examples]

    batch = {
        "original_size": torch.tensor(original_size),
        "bucket_id": torch.tensor(examples[0]["bucket_id"]),
        "gly_line": gly_line,
        "crops_coords_top_left": torch.tensor(crops_coords_top_left),
        "glyphs": glyphs,
        "positions": positions,
        "pixel_values": pixel_values,
        "instance_prompt": instance_prompt,    
        "instance_prompt_glyph": instance_prompt_glyph,       
        "input_ids": torch.cat(input_ids),
        "input_ids_uncond": torch.cat(input_ids_uncond),
        "n_lines": n_lines,
    }

    return batch


if __name__ == '__main__':

    urls=["/mnt/data/group/text2img_data/data_font/haibao/1/{00001..00020}.tar",\
        "/mnt/data/group/text2img_data/data_font/2_tusij_process/*/* ",\
        "/mnt/data/group/text2img_data/data_font/3_wukong_process/*/*"] 

    all_urls = []
    for url in urls:
        if "*" in url:
            all_urls += expand_urls1(url)
        else:
            all_urls += expand_urls(url)
    import open_clip
    tokenizer = open_clip.get_tokenizer('xlm-roberta-large-ViT-H-14')

    ds = ImageEmbeddingDataset(
        all_urls,
        tokenizer,
        shuffle_shards=False,
        resample=False,
        handler=wds.handlers.warn_and_continue
    )

    source_dp = IterableWrapper(ds)

    def split_bucket(n):
        return n["bucket_id"]

    dp_list = source_dp.mydemux(num_instances=len(BUCKETS), classifier_fn=split_bucket,buffer_size=1000)
    pipes_to_weights_dict = {}

    for i in range(len(dp_list)):
        pipes_to_weights_dict[dp_list[i]] = BUCKET_PROBS[i]
    sample_mul_dp = SampleMultiplexer(pipes_to_weights_dict=pipes_to_weights_dict, batch_size=1, seed=0).collate(collate_fn=collate_fn)
    mp_rs = MultiProcessingReadingService(num_workers=1)
    dl = DataLoader2(sample_mul_dp, reading_service=mp_rs)
    fw = open("tmp1.txt","w")
    for i, batch in enumerate(tqdm(dl)):
        if i<10:
            for j,(t,image,png,positions) in enumerate(zip(batch["instance_prompt_glyph"],batch["glyphs"],batch["pixel_values"],batch["positions"])):
                fw.write(t[0]+"\n")
                save_image(image,f"tmp/{i}_{j}.jpg")
                save_image(png,f"tmp/{i}_{j}.png")
                save_image(positions,f"tmp/{i}_{j}_p.png")

