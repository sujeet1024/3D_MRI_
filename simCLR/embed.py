import argparse
import torch
import pandas as pd
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, BrainDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR, ResFeat

import os
import numpy as np
import pandas as pd
import nibabel as nib
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm

import csv

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda:0')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    ###################
    ###################
    paths = pd.read_csv('../val_PPMIs.csv')
    setso = 'Val'
    out_path = f'./Embeds/{setso}/'
    device = 'cuda:0'
    ###################
    ###################
    train_dataset = BrainDataset(paths, args.device, 1) #args.n_views)
    # train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=False, drop_last=True)       # num_workers=0, 

    # model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    r3d_18 = models.video.r3d_18(pretrained=True)
    # Change the first layer
    r3d_18.stem[0] = torch.nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
    model = ResFeat(r3d_18)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        checkpoint = torch.load('/media/data/3drepr/simCLR/runs/Run02_2/checkpoint_0050.pth.tar')			# ('SimCLR200.pth')
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        # simclr.load_state_dict(checkpoint)
        # simclr.load_state_dict(checkpoint['state_dict'])			# ('SimCLR200.pth')
        # simclr.train(train_loader)
        # simclr.eval()
        for i in tqdm(range(len(paths))):
            example = np.array(nib.load(os.path.abspath(paths.iloc[i,0])).dataobj, dtype=np.float32)
            labelf = paths.iloc[i,2]
            if example.shape[0]>256:
                if example.shape[1]==512:
                    if example.shape[2]<208:
                        example = pad0(example, 386)
                    example = pool3d(example[60:-64,60:-64,:],6,6)
                else:
                    if example.shape[2]<208:
                        example = pad0(example)
                    example = pool3d(example[34:226,29:221,19:211],3,3)
            else:
                if labelf==0:
                    if example.shape[2]<208:
                        example = pad0(example)
                        example = pool3d(example[34:226,29:221,19:211],3,3)
                    elif example.shape[2]==248:
                        example = pool3d(example[34:226,29:221,19:211],3,3)
                else:
                    if example.shape[2]<208:
                        if example.shape[1]<210:
                            example =pad0(example)
                            example = pool3d(example[:,:,19:211],3,3)
                        else:
                            example = pad0(example)
                            example = pool3d(example[38:232,29:221,19:211],3,3)
                    elif example.shape[2]==248:
                        example = pool3d(example[32:224,29:221,19:211],3,3)
            example = np.swapaxes(example, 0,1)
            example = np.flip(example, axis=1)
            example = normz(example, is_torch=False)
            example = np.expand_dims(example, axis=0)
            example = np.expand_dims(example, axis=0)
            if example.shape==(1,1,64,64,64):
                example = torch.from_numpy(example).to(device)
                example, _ = simclr(example)
                example = example.detach().cpu().numpy()
                start_index = paths.iloc[i,0].rfind('/')+1
                save_path = f'{out_path}{paths.iloc[i,0][start_index:-7]}.npy'
                np.save(save_path, example)
                with open (f'annotations_{setso}_PPMIs.csv', 'a', newline='') as anns:
                    row_data = [os.path.abspath(save_path), 'PPMI', labelf]
                    writer = csv.writer(anns)
                    writer.writerow(row_data)
        
        # embed, _ = simclr(example)








def pool3d(A, kernel_size, stride, padding=0, pool_mode='avg'):
    '''
    3D Pooling

    Parameters:
    A: input 3D array
    kernel_size: int, the size of the window over which we take pool
    stride: int, the stride of the window
    padding: int, implicit zero paddings on both sides of the input
    pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    # A = np.pad(A, padding, mode='constant')
    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1,
                    (A.shape[2] - kernel_size)//stride + 1)
    shape_w = (output_shape[0], output_shape[1], output_shape[2], kernel_size, kernel_size, kernel_size)
    strides_w = (stride*A.strides[0], stride*A.strides[1], stride*A.strides[2], A.strides[0], A.strides[1], A.strides[2])
    A_w = as_strided(A, shape_w, strides_w)
    # Return the result of pooling
    # print('Pooled/n')
    if pool_mode == 'max':
        return A_w.max(axis=(3, 4, 5))
    elif pool_mode == 'avg':
        return A_w.mean(axis=(3, 4, 5))
    

def avgpool3dnp(x, kernel_s, stride):
    x_new = np.zeros((x.shape[0]//stride, x.shape[1]//stride, x.shape[2]//stride))
    for i in range(x_new.shape[0]):
        for j in range(x_new.shape[1]):
            for k in range(x_new.shape[2]):
                x_new[i,j,k] = np.mean(x[i*stride:i*stride+kernel_s, j*stride:j*stride+kernel_s, k*stride:k*stride+kernel_s])
    return x_new


def normz(x, normz_val=1, is_torch=True):
    if is_torch:
        return (x-torch.min(x))/(torch.max(x)-torch.min(x))*normz_val
    else: return (x-np.min(x))/(np.max(x)-np.min(x))*normz_val



def pad0(x, out_shape:int=212):
    in_shape = x.shape[2]
    front = (out_shape-in_shape)//2-1
    back = (out_shape-in_shape)//2+1
    padsf = np.zeros((x.shape[0], x.shape[1], front), dtype=np.float32)
    padsb = np.zeros((x.shape[0], x.shape[1], back), dtype=np.float32)
    return np.concatenate((padsf, x, padsb), axis=2)




if __name__ == "__main__":
    main()
