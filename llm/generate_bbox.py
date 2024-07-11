# Copyright (c) OPPO Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import braceexpand
import webdataset as wds
from tqdm import tqdm
import random
import numpy as np
import os,glob
from transformers import AutoTokenizer

from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import re
from PIL import Image
from PIL import ImageDraw

USED_KEYS = ["txt","jpg","text","json"]

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


def verify_keys(samples, required_keys, handler=wds.handlers.reraise_exception):
    """
    Requires that both the image and embedding are present in the sample
    This is important to do as a user may forget they do not have embeddings in their webdataset and neglect to add them using the embedding_folder_url parameter.
    """

    for sample in samples:
        sample_json = sample["json"]
        fonts = "".join([t[0] for t in sample_json["font"]])
        if len(fonts)<5:continue
        is_normal = True
        for key in required_keys:
            if key not in sample:
                print(f"Sample {sample['__key__']} missing {key}. Has keys {sample.keys()}")
                is_normal = False
        if is_normal:
            yield {key: sample[key] for key in required_keys}
key_verifier = wds.filters.pipelinefilter(verify_keys)

def get_caption(ori_caption, text_list, width, height):
    new_caption = ori_caption + random.choice(phrase_list)
    pos=''
    if random.random() > 0.2:
        is_comma = True
    else:
        is_comma = False
    for text in text_list:
        if is_comma:
            text = "“" + text + "”"
        pos += text + ' , '
    pos = pos[:-2] + '。'
    pos += f"原图的宽和高:{width,height}"
    new_caption += pos
    return new_caption

class ImageEmbeddingDataset(wds.DataPipeline, wds.compat.FluidInterface):
    """
    A fluid interface wrapper for DataPipline that retontsurns image embedding pairs
    Reads embeddings as npy files from the webdataset if they exist. If embedding_folder_url is set, they will be inserted in from the alternate source.
    """
    def __init__(
            self,
            urls,
            handler=wds.handlers.reraise_exception,
            resample=False,
            shuffle_shards=True,
    ):

        super().__init__()
        keys = USED_KEYS 
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

    def preproc(self, sample):
        """Applies the preprocessing for prompts"""
        example = {}
        instance_image = sample["jpg"]
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        width, height = instance_image.size
        parsed_data = sample["json"]
        font_data = parsed_data["font"][:15]
        polygons = []
        polygons_ori = []
        texts = []
        for item in font_data:
            coordinates = item[1].split()
            polygon = []
            for coordinate in coordinates:
                x, y = tuple(map(int, re.findall(r'\d+', coordinate)))
                polygon.append([round(x/width, 2), round(y/height, 2)])
            polygon_diagonal = [polygon[0],polygon[2]]
            delta_x = polygon_diagonal[1][0]-polygon_diagonal[0][0]
            delta_y = polygon_diagonal[1][1]-polygon_diagonal[0][1]
            if delta_x>0.05 and delta_y>0.05:
                polygon_diagonal_ori = [list(eval(coordinates[0])),list(eval(coordinates[2]))]
                polygons.append(polygon_diagonal)
                polygons_ori.append(polygon_diagonal_ori)
                texts.append(item[0])

        output = str(dict(zip(texts,polygons)))
        output_ori = str(dict(zip(texts,polygons_ori)))
        count=0
        if '*' in sample["txt"]:
            count += 1
            sample["txt"] = sample["txt"].replace('*', " ")
        if count > 0:
            print(f"Found {count} image's caption contain placeholder: *, change to ' '...")

        new_caption = get_caption(sample["json"]["caption_zh"], texts, width, height)
        inputs = new_caption

        # example['instruction'] =  instruction
        example['input'] = inputs
        example['output'] = output
        example['output_ori'] = output_ori

        return example


def batch_collate_fn(examples):
    instruction = [example["instruction"] for example in examples]
    inputs = [example["input"] for example in examples]
    output = [example["output"] for example in examples]
    output_ori = [example["output_ori"] for example in examples]

    batch = {
        "instruction": instruction,
        "input": inputs,
        "output": output,
        "output_ori": output_ori
    }

    return batch


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class DataModuleCustom(LightningDataModule):
    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Universal DataModule')
        parser.add_argument('--webdataset_base_urls', type=str, nargs="+")
        parser.add_argument('--num_workers', default=2, type=int)
        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--shard_width', default=5, type=int)
        parser.add_argument('--hr_size', default=-1, type=int)
        parser.add_argument('--train_split', default=1.0, type=float)
        parser.add_argument('--val_split', default=0.0, type=float)
        parser.add_argument('--test_split', default=0.0, type=float)
        parser.add_argument('--shuffle_train', default=False, action="store_true")
        parser.add_argument('--resample_train', default=False, action="store_true")
        parser.add_argument('--shuffle_num', default=None, type=int)
        parser.add_argument('--test_prompts', type=str, default="./test_prompts.txt")
        parser.add_argument('--test_repeat', default=1, type=int)

        parser.add_argument(
            "--resolution", type=int, default=512,
            help=(
                "The resolution for input images, all the images in the train/validation dataset will be resized to this"
                " resolution"
            ),
        )
        parser.add_argument(
            "--center_crop", action="store_true", default=True,
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
        self.collate_fn = batch_collate_fn
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
            all_urls += expand_urls(url)
        num_train = round(self.train_prop*len(all_urls))
        num_test = round(self.test_prop*len(all_urls))
        num_val = len(all_urls) - num_train - num_test
        assert num_train + num_test + num_val == len(all_urls), f"{num_train} + {num_test} + {num_val} = {num_train + num_test + num_val} != {len(all_urls)}"
        self.train_urls, self.test_urls, self.val_urls = random_split(all_urls, [num_train, num_test, num_val])  # , generator=torch.Generator().manual_seed(self.seed)
        
    def setup(self, stage=None):
        if 'train' in self.datasets:
            self.datasets['train'] = ImageEmbeddingDataset(
                self.train_urls,
                self.tokenizer,
                shuffle_shards=self.shuffle_train,
                resample=self.resample_train,
                handler=wds.handlers.warn_and_continue,
            )
            if self.shuffle_num is not None and self.shuffle_num > 0:
                self.datasets['train'].shuffle(self.shuffle_num)

    def _train_dataloader(self):
        # return self.create_dataloader(self.train_urls, shuffle=self.shuffle_train, resample=self.resample_train)
        return DataLoaderX(
            self.datasets['train'],
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            prefetch_factor=2,  # This might be good to have high so the next npy file is prefetched
            pin_memory=True,
            shuffle=False,
            worker_init_fn=None,
            collate_fn=self.collate_fn,
        )


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

if __name__ == '__main__':
    instruction = "下面是描述任务的指令，并进一步提供上下文的输入配对。生成可以满足需求的回答。 \
            背景：用户输入一句话，其中意图是想画一张带有文字的海报。其中一句话包含要渲染的字的信息。\
            步骤1：识别出用户作画时想要在图片中渲染的文字。需要渲染的文字前一般会有一些提示词，比如：文本的内容是; 图像中描绘的文本材料是; 文本说; 上面写着; 快照中显示的标题是，等等。 或者是引号“”中的内容。 \
            步骤2：根据步骤1每一块渲染字的个数和常规海报或者设计知识，估计每一块渲染的字的具体的坐标点，每个坐标点都要经过标准化并且必须在[0,1]区间中。 \
            注意：步骤2我们只需要给出两个坐标点，分别是左上角和右下角。例如(x1, y1),(x2, y2)这两个坐标点，其中x2-x1代表的比例长度就是文字框的长度，y2-y1代表的比例长度就是文字框的宽度；\
            如果文字横向排列，则x2-x1代表的长度会和字的个数正比,y2-y1代表的长度就是字的宽度；如果文字纵向排列,则正好相反。字的大小会根据海报布局而发生变化。\
            另外有可能两个连续的文字框是连续的内容，所以x坐标或者y坐标是相同的，表示文字框挨着。文字框坐标应该能够准确地框选出文本内容所在的区域，并且与渲染文字快一一对应。"

    instruction1 = "下面是描述任务的指令，并进一步提供上下文的输入配对。生成可以满足需求的回答。 \
            背景：用户输入一句话，其中意图是想画一张带有文字的海报。其中一句话包含要渲染的字的信息。\
            步骤1：识别出用户作画时想要在图片中渲染的文字。需要渲染的文字前一般会有一些提示词，比如：文本的内容是; 图像中描绘的文本材料是; 文本说; 上面写着; 快照中显示的标题是，等等。 或者是引号“”中的内容。 \
            步骤2：根据步骤1每一块渲染字的个数和常规海报或者设计知识，估计每一块渲染的字的具体的坐标点，每个坐标点都要经过标准化并且必须在[0,1]区间中。 \
            注意：步骤2我们只需要给出两个坐标点，分别是左上角和右下角。例如(x1, y1),(x2, y2)这两个坐标点，其中x2-x1代表的比例长度就是文字框的长度，y2-y1代表的比例长度就是文字框的宽度；\
            如果文字横向排列，则x2-x1代表的长度会和字的个数正比,y2-y1代表的长度就是字的宽度；如果文字纵向排列,则正好相反。字的大小会根据海报布局而发生变化。\
            另外有可能两个连续的文字框是连续的内容，所以x坐标或者y坐标是相同的，表示文字框挨着。文字框坐标应该能够准确地框选出文本内容所在的区域，并且与渲染文字快一一对应。"


    urls=[
       "/mnt/data/group/text2img_data/data_font/data_scraping_1/{00000..03638}.tar",
       "/mnt/data/group/text2img_data/data_font/data_scraping_0/{00000..04515}.tar" ,
       "/mnt/data/group/text2img_data/data_font/data_scraping_2023/{00000..01023}.tar" ,
       "/mnt/data/group/text2img_data/data_font/data_scraping_theme/{00000..04770}.tar" ,
       "/mnt/data/group/text2img_data/data_font/laion/{00000..14339}.tar"
        ] 

    all_urls = []
    for url in urls:
        if "*" in url:
            all_urls += expand_urls1(url)
        else:
            all_urls += expand_urls(url)
    print(len(all_urls))

    ds = ImageEmbeddingDataset(
        all_urls,
        shuffle_shards=True,
        resample=False,
        handler=wds.handlers.warn_and_continue
    )
    
    text_input_format=[]
    with open("data.json", "w", encoding="utf-8") as f:
        for item in tqdm(iter(ds)):
            if item["output"]=='{}':continue
            # item["instruction"] = instruction
            json.dump(item, f, ensure_ascii=False)
            text_input_format.append(item)
            f.write("\n")