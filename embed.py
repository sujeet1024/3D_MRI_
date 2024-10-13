import torch
import pandas as pd
from Models import *

import os
import numpy as np
import pandas as pd
import nibabel as nib
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm

import csv

seed = 0

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



###########
###########
latent_dim=1000
annotations_file = pd.read_csv('./train_PPMIs.csv', header=None)
annotations_val = pd.read_csv('./val_PPMIs.csv', header=None)
annotations_test = pd.read_csv('./test_PPMIs.csv', header=None)

paths = annotations_test
setso = 'Test'
out_path = f'./checkpoint51/Embeds/{setso}/'
###########
###########


E = Discriminator(out_class = latent_dim,is_dis=False)
E.load_state_dict(torch.load(f'./checkpoint{51}/E_iter{940}es.pth'))
E.to(device)
E.eval()


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
        example = E(example)
        example = example.detach().cpu().numpy()
        start_index = paths.iloc[i,0].rfind('/')+1
        save_path = f'{out_path}{paths.iloc[i,0][start_index:-7]}.npy'
        np.save(save_path, example)
        with open (f'./checkpoint51/annotations_{setso}_PPMIs.csv', 'a', newline='') as anns:
            row_data = [os.path.abspath(save_path), 'PPMI', labelf]
            writer = csv.writer(anns)
            writer.writerow(row_data)