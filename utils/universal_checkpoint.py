# Copyright (c) OPPO Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from pytorch_lightning.callbacks import ModelCheckpoint
import os


class UniversalCheckpoint(ModelCheckpoint):
    @staticmethod
    def add_argparse_args(parent_args):
        parser = parent_args.add_argument_group('universal checkpoint callback')

        parser.add_argument('--monitor', default='train_loss', type=str)
        parser.add_argument('--mode', default='min', type=str)
        parser.add_argument('--save_ckpt_path', default='./ckpt/', type=str)
        parser.add_argument('--filename', default='model-{epoch:02d}-{train_loss:.4f}', type=str)
        parser.add_argument('--save_last', action='store_true', default=False)
        parser.add_argument('--save_top_k', default=3, type=float)
        parser.add_argument('--every_n_train_steps', default=None, type=float)
        parser.add_argument('--save_weights_only', action='store_true', default=False)
        parser.add_argument('--every_n_epochs', default=None, type=int)
        parser.add_argument('--save_on_train_epoch_end', action='store_true', default=None)
        parser.add_argument('--load_ckpt_id',default=0, type=int)
        parser.add_argument('--load_ckpt_path', default='/mnt/data/group/majian/glyphdraw2/checkpoints/sdxl_mul_all', type=str)
        parser.add_argument('--align_loss', default=True, type=bool)
        parser.add_argument('--controlnet_condition', default=True, type=bool)
        parser.add_argument('--every_n_steps', default=5000, type=int)
        parser.add_argument('--proj_path', default="/mnt/data/group/majian/text2img_oppo_model/general_v3_model/mulclip_v3/proj_0_109999/pytorch_model.bin", type=str)

        return parent_args

    def __init__(self, args):
        super().__init__(monitor=args.monitor,
                         save_top_k=args.save_top_k,
                         mode=args.mode,
                         every_n_train_steps=args.every_n_train_steps,
                         save_weights_only=args.save_weights_only,
                         dirpath=args.save_ckpt_path,
                         filename=args.filename,
                         save_last=args.save_last,
                         every_n_epochs=args.every_n_epochs,
                         save_on_train_epoch_end=args.save_on_train_epoch_end)

        if args.load_ckpt_path is not None and \
                not os.path.exists(args.load_ckpt_path):
            print('--------warning no checkpoint found--------, remove args')
            args.load_ckpt_path = None
