import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
import pickle
import time
import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from timm.optim import optim_factory

# import post_training_utils
from vit_ae.misc import NativeScalerWithGradNormCount as NativeScaler
from vit_ae.misc import BrainDataset
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

import vit_ae.train_one_epoc as train_one_epoc
# from vit_ae.train_3d_resnet import get_all_feat_and_labels
# from dataset.dataset_factory import get_dataset
from vit_ae_env import PROJECT_ROOT_DIR
from vit_ae.model_factory import get_models
from vit_ae.vit_helpers import interpolate_pos_embed
from vit_ae.read_configs import bootstrap
from vit_ae import misc
import torchio as tio

# from utils.feature_extraction import generate_features


def get_args_parser():
    parser = argparse.ArgumentParser('K-fold cross validation', add_help=False)

    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    # Model parameters
    parser.add_argument('--model', default='contr_mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',  # earlier 0
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--dist_on_itp', action='store_true')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--device', default='cuda:1',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')

    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    return parser



def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    # device = torch.device("cuda:0")

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    transforms = [
        tio.RandomAffine(),
        tio.RandomNoise(std=0.1),
        tio.RandomGamma(log_gamma=(-0.3, 0.3))
    ]
    args = bootstrap(args=args, key='K_FOLD')
    train_transforms = tio.Compose(transforms)
    print(f"Masking ratio is {args.mask_ratio}")

    log_dir = os.path.join(PROJECT_ROOT_DIR, args.log_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=log_dir)

    split_index_path = os.path.join(PROJECT_ROOT_DIR, "brats", 'k_fold', 'indices_file')
    os.makedirs(split_index_path, exist_ok=True)

    Train_files = pd.read_csv('/home/guest1/data/3drepr/train_ADMIC.csv', header=None)
    # Test_files = pd.read_csv('/home/guest1/data/3drepr/test_ADNI.csv', header=None)
    trainset = BrainDataset(Train_files, transform=train_transforms, device=device)
    # testset = BrainDataset(Test_files, device=device)
    # Needed for the pre-training phase
    args.nb_classes = 2
    data_loader_train = torch.utils.data.DataLoader(
        trainset,              # dataset_whole, sampler=train_subsampler,
        batch_size=args.batch_size,
        num_workers=0,          # args.num_workers,
        pin_memory=False,       # args.pin_mem,
        drop_last=True,
    )
    model = get_models(model_name='autoenc', args=args)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    args.output_dir = os.path.join(PROJECT_ROOT_DIR, args.output_dir, 'checkpoints')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Start training for {args.epochs} epochs")

    start_time = time.time()
    min_loss = float('inf')
    for epoch in range(args.start_epoch, args.epochs):
        # loss weighting for the edge maps
        if not args.use_edge_map:
            edge_map_weight = 0
        else:
            edge_map_weight = 0.01 * (1 - epoch / args.epochs)
        train_stats = train_one_epoc.train_one_stage_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
            edge_map_weight=edge_map_weight
        )

        if train_stats['loss'] < min_loss:
            min_loss = train_stats['loss']
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=f"min_loss_k_fold_split_{epoch}")

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # Now we would go ahead and also do the feature extraction for this split
    del model
    torch.cuda.empty_cache()
    # Starting the extraction process
    model = get_models(model_name='vit', args=args)
    args.log_dir = os.path.join(PROJECT_ROOT_DIR, args.log_dir)

    args.finetune = os.path.join(args.output_dir, f"checkpoint-min_loss_k_fold_split_{0}.pth")
    checkpoint = torch.load(args.finetune, map_location='cpu')

    print("Load pre-trained checkpoint from: %s" % args.finetune)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    model.to(device)

    if args.global_pool:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    else:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, "brats", 'ssl_features_dir', args.subtype)
    os.makedirs(ssl_feature_dir, exist_ok=True)
    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
