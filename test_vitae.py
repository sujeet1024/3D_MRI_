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
from torch.autograd import Variable
from einops import rearrange
from sklearn.model_selection import StratifiedKFold
from skimage.metrics import structural_similarity as ssim
from timm.optim import optim_factory

# import post_training_utils
from vit_ae.misc import NativeScalerWithGradNormCount as NativeScaler
from vit_ae.misc import BrainDataset, ssim_eval
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

    parser.add_argument('--device', default='cuda:0',
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
    device = torch.device(args.device)
    print(device, 'experiment: {exp_no} with vit')
    torch.manual_seed(0)
    np.random.seed(0)

    transforms = [
        tio.RandomAffine(),
        tio.RandomNoise(std=0.1),
        tio.RandomGamma(log_gamma=(-0.3, 0.3))
    ]
    args = bootstrap(args=args, key='K_FOLD')
    train_transforms = tio.Compose(transforms)

    Test_files = pd.read_csv('/home/guest1/3dmri/3dbraingen/test_AOMIC.csv', header=None)
    testset = BrainDataset(Test_files, device=device)

    data_loader_test = torch.utils.data.DataLoader(
        testset,              # dataset_whole, sampler=train_subsampler,
        batch_size=args.batch_size,
        num_workers=0,          # args.num_workers,
        pin_memory=False,       # args.pin_mem,
        drop_last=True,
    )
    args.volume_size=(64,64,64)
    model = get_models(model_name='autoenc', args=args)

    model.to(device)
    model_without_ddp = model


    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    loss_scaler = NativeScaler()


    # misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    args.output_dir = os.path.join(PROJECT_ROOT_DIR, 'checkpoints_vit')
    os.makedirs(args.output_dir, exist_ok=True)

    min_loss = float('inf')
    print('epochs: ', args.epochs)


    # for epoch in range(args.start_epoch, args.epochs):
    args.finetune = os.path.join(args.output_dir, f"checkpoint-min_loss_vit.pth")
    if os.path.exists(args.finetune):
        checkpoint = torch.load(args.finetune, map_location='cpu')
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

        model.to(device)
        model.eval()
        ssim_avgs = 0

        with open(os.path.join(args.output_dir, "test.txt"), mode="a", encoding="utf-8") as f:
                f.write(f"This results are for the experiment on ViTAE++ trained on ADMIC\n\n")

        for i, (real_images, label) in enumerate(data_loader_test):
            real_images = (Variable(real_images, requires_grad=False).type(torch.cuda.FloatTensor)).cuda(device, non_blocking=True)
            label = (Variable(label, requires_grad=False).type(torch.cuda.FloatTensor)).cuda(device, non_blocking=True)
            loss, x_hat, mask, p1, p2, z1, z2 = model(real_images, label)
            x_hat = model.unpatchify(x_hat)
            # print('\n*******', real_images.shape)
            # print('\n*******', label.shape)
            # print('\n*******', x_hat.shape)
            ssim_score = ssim_eval(x_hat, label)
            ssim_avgs += ssim_score
            with open(os.path.join(args.output_dir, "test.txt"), mode="a", encoding="utf-8") as f:
                f.write(f"[{i}/200000] \t\t ssim: {ssim_score:<8.5}\n")
        ssim_avgs = ssim_avgs/len(data_loader_test)
        with open(os.path.join(args.output_dir, "test.txt"), mode="a", encoding="utf-8") as f:
            f.write(f"averages to {ssim_avgs:<8.5} for epoch 220\n\n\n")



def ssim_eval(img, gen):
    img = torch.round(normz(img, 255))
    gen = torch.round(normz(gen, 255))
    img = img.squeeze().cpu().detach().numpy()
    gen = gen.squeeze().cpu().detach().numpy()
    window_size = min(img.shape[0]//2*2-1, img.shape[1]//2*2-1, img.shape[2]//2*2-1, 7)
    # ssim_noise = ssim(img.squeeze().cpu().numpy(), gen.squeeze().cpu().numpy(), data_range=255)    #img_noise.max() - img_noise.min())
    ssim_noise = ssim(img, gen, data_range=255, win_size=window_size)    #img_noise.max() - img_noise.min())
    return ssim_noise

def normz(x, normz_val=1, is_torch=True):
    if is_torch:
        return (x-torch.min(x))/(torch.max(x)-torch.min(x))*normz_val
    else: return (x-np.min(x))/(np.max(x)-np.min(x))*normz_val





if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
