# GlyphDraw2
[[Paper](https://arxiv.org/abs/2407.02252)]


## Requirements
A suitable [conda](https://conda.io/) environment named `glyphdraw2` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate glyphdraw2
```

## Training 


```bash
bash train_glyphdraw2.sh 0 8
```
The first parameter represents The serial number of the current process, used for inter process communication. The host with rank=0 is the master node.
and the second parameter the world size.Please review the detailed parameters of model training
with train_glyphdraw2.sh script

## Inference

We provide one script to generate images using checkpoints. Include Clip checkpoints, GlyphDraw2 checkpoints. Then run
```bash
python test_glyphdraw2.py --proj_path=path_to_PEA_proj --text_proj_path=path_to_GlyphDraw2_text_proj --ctrlnet_path=path_to_GlyphDraw2_ControlNet --unet_path=path_to_GlyphDraw2_unet
```
You can check `test_glyphdraw2.py` for more details about interface. 


## Training data preprocessing
Please refer to the `data_process` directory, the paths needed in the .py script need to be downloaded independently. Additionally, if [LaMa](https://github.com/advimman/lama)  processed data is required, you will have to download the source code of saicinpainting yourself. The data reading format is webdataset, and the output format is the training data format of Glyphdraw2.

## Fine-tuning LLM
We fine-tune LLM based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). All you need to do is process the training data and modify the yaml. For relevant configuration and data processing code, please refer to the `llm` directory.


## Acknowledgements 
This code is builds on the code from the [GlyphDraw](https://github.com/OPPO-Mente-Lab/GlyphDraw) library. 
we borrow some code from [AnyText](https://github.com/tyxsspa/AnyText), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter),  [Fengshenbang](https://github.com/IDEA-CCNL/Fengshenbang-LM), [Subject-Diffusion](https://github.com/OPPO-Mente-Lab/Subject-Diffusion), and [PEA-Diffusion](https://github.com/OPPO-Mente-Lab/PEA-Diffusion). Sincere thanks!
