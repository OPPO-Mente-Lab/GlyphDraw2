#!/bin/bash
pip install einops_exts rotary_embedding_torch easydict scikit-image Pillow==9.5
cp /mnt/data/group/majian/glyphdraw2/font/*.ttf /usr/share/fonts/truetype/dejavu/
sudo chmod 777 /usr/share/fonts/truetype/dejavu/*.ttf
ROOT_DIR=./checkpoints

MODEL_NAME=sdxl_mul_all    # https://odocs.myoas.com/docs/Ee32M7ad8Wfo0A2J
MODEL_ROOT_DIR=$ROOT_DIR/${MODEL_NAME}
if [ ! -d ${MODEL_ROOT_DIR} ];then
  mkdir ${MODEL_ROOT_DIR}
fi

MICRO_BATCH_SIZE=2

CONFIG_JSON="$MODEL_ROOT_DIR/${MODEL_NAME}.ds_config.json"
ZERO_STAGE=2
cat <<EOT > $CONFIG_JSON
{
    "zero_optimization": {
        "stage": ${ZERO_STAGE}
    },
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE
}
EOT
export PL_DEEPSPEED_CONFIG_PATH=$CONFIG_JSON


DATA_ARGS="\
        --webdataset_base_urls \
        /mnt/data/group/text2img_data/data_font/data_scraping_1/{00000..03638}.tar \
       /mnt/data/group/text2img_data/data_font/data_scraping_0/{00000..04515}.tar \
       /mnt/data/group/text2img_data/data_font/data_scraping_2023/{00000..01023}.tar \
       /mnt/data/group/text2img_data/data_font/data_scraping_theme/{00000..04770}.tar \
       /mnt/data/group/text2img_data/data_font/laion/{00000..14339}.tar \
       /mnt/data/group/text2img_data/data_font/wokong/{00000..10315}.tar \
        /mnt/data/group/text2img_data/data_font/1_50W_process/*/* \
        /mnt/data/group/text2img_data/data_font/2_tusij_process/*/* \
        /mnt/data/group/text2img_data/data_font/3_wukong_process/*/* \
        /mnt/data/group/text2img_data/data_font_en/ae/*/* \
        /mnt/data/group/text2img_data/data_font_en/BLIP_tar_512/*/* \
        /mnt/data/group/text2img_data/data_font_en/coyo/*/* \
        /mnt/data/group/text2img_data/data_font_en/coyo1/*/* \
        /mnt/data/group/text2img_data/data_font_en/laion2b/*/* \
        /mnt/data/group/text2img_data/data_font_en/laion400/*/* \
        /mnt/data/group/text2img_data/data_font_en/laion_mul/*/* \
        --num_workers 2 \
        --batch_size $MICRO_BATCH_SIZE \
        --train_split 1.0 \
        --val_split 0.0 \
        --test_split 0.0 \
        --resample_train \
        "

MODEL_ARGS="\
        --model_path /mnt/data/group/majian/text2img_oppo_model/stable-diffusion-xl-protovisionXLv6 \
        --chinese_clip_path /mnt/data/group/majian/text2img_oppo_model/general_v3_model/open_clip_pytorch_model.bin \
        --learning_rate 1e-5 \
        --weight_decay 1e-1 \
        --warmup_steps 100 \
        "

MODEL_CHECKPOINT_ARGS="\
        --save_last \
        --save_ckpt_path ${MODEL_ROOT_DIR}/ckpt \
        "

TRAINER_ARGS="\
        --max_epoch 10 \
        --accelerator gpu \
        --devices -1 \
        --num_nodes $WORLD_SIZE \
        --strategy deepspeed_stage_${ZERO_STAGE} \
        --log_every_n_steps 100 \
        --precision 16 \
        --default_root_dir ${MODEL_ROOT_DIR} \
        --replace_sampler_ddp False \
        --num_sanity_val_steps 0 \
        --limit_val_batches 10 \
        --accumulate_grad_batches 1 \
        "

export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "


export CC=gcc
export CXX=g++
python -m torch.distributed.run \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=8 \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    train_glyphdraw2.py $options
