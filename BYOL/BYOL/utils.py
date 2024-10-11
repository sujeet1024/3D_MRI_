import os
import torch
import torch.nn as nn
import torchio as tio
import numpy as np
import nibabel as nib

from torch.utils.data import DataLoader, Dataset
from numpy.lib.stride_tricks import as_strided

class EMA(object):
    """
    Exponential Moving Average class.
    
    :param beta: float; decay parameter
    """
    def __init__(self, beta):
        self.beta = beta
        
    def __call__(self, MA, value):
        return MA * self.beta + (1 - self.beta) * value
    

class RandomApply(nn.Module):
    """
    Randomly apply function with probability p.
    
    :param func: function; takes input x (likely augmentation)
    :param p: float; probability of applying func
    """
    def __init__(self, func, p):
        super(RandomApply, self).__init__()
        self.func = func
        self.p = p
        
    def forward(self, x):
        if torch.rand(1)  > self.p:
            return x
        else:
            return self.func(x)
        

class Hook():
    """
    A simple hook class that returns the output of a layer of a model during forward pass.
    """
    def __init__(self):
        self.output = None
        
    def setHook(self, module):
        """
        Attaches hook to model.
        """
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        """
        Saves the wanted information.
        """
        self.output = output
        
    def val(self):
        """
        Return the saved value.
        """
        return self.output
    




class EmbedSet(Dataset):
    def __init__(self, annotations_file, device):
        self.annotations = annotations_file
        self.device = device

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        example_path, labelf = os.path.abspath(self.annotations.iloc[idx, 0]), self.annotations.iloc[idx, 2]
        example = np.load(example_path)
        example = torch.squeeze(torch.from_numpy(example))
        example = example.to(self.device)
        return example, labelf



class BrainDataset(Dataset):
    def __init__(self, annotations_file, device, n_views:int=0, transform=None, target_transform=None):
        self.annotations = annotations_file #load csv file containing training examples using pandas
        # self.transform = ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(64), n_views)
        transforms = [
            tio.RandomAffine(),
            tio.RandomNoise(std=0.1),
            tio.RandomGamma(log_gamma=(-0.3, 0.3))
        ]
        train_transforms = tio.Compose(transforms)
        self.transform = train_transforms
        self.target_transform = target_transform
        # self.orig_transform = ContrastiveLearningViewGenerator(self.get_simclr_pipeline_trans_orig(64), n_views)
        self.device = device

    def __len__(self):
        return len(self.annotations)
    
    # @staticmethod
    # def get_simclr_pipeline_trans_orig(size, s=1):
    #     data_transforms = tio.Compose([])
    #     return data_transforms

    # @staticmethod
    # def get_simclr_pipeline_transform(size, s=1):
    #     """Return a set of data augmentation transformations as described in the SimCLR paper."""
    #     # color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    #     data_transforms = tio.Compose([tio.RandomFlip(),
    #                                    tio.RandomSwap(4,4),
    #                                    tio.RandomBlur(),
    #                                    tio.RandomGamma()])
    #     return data_transforms

    def __getitem__(self, idx):
        example_path, dataset_name = os.path.abspath(self.annotations.iloc[idx, 0]), self.annotations.iloc[idx,1]
        # load example into numpy array using nibabel
        example = nib.load(example_path)
        example = np.array(example.dataobj, dtype=np.float32)
        if dataset_name=='AOMIC':
            example = example[:, 23:215, 10:202]
            example = pad0(example)
            example = pool3d(example, 3, 3)
        elif dataset_name=='ADNI':
            example = pool3d(example[34:226,29:221,19:211],3,3)
            example = np.swapaxes(example, 1,2)
            example = np.flip(example, axis=2)
        elif dataset_name=='ABIDE':
            # example = np.array(example.get_fdata(), dtype=np.float32)
            example = pad0(example, 1, 2)
        elif dataset_name=='PPMI':
            labelf = self.annotations.iloc[idx,2]
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
            example = np.swapaxes(example, 0,2)
            # example = np.swapaxes(example, 1,2)
            example = np.flip(example, axis=2)
        example = normz(example, is_torch=False)
        example = np.expand_dims(example, axis=0)
        if example.shape!=(1,64,64,64):
            print(example_path)
        # print(example.shape, "pre_transforms")
        label = example.copy()
        # print(label.shape)
        if self.transform:
            # orig = self.transform(example)
            # orig = [torch.from_numpy(i) for i in orig]
            # orig = [i.to(self.device) for i in orig]
            example = self.transform(example)
            example = [torch.from_numpy(i) for i in example] # , dtype=torch.float32
            example = [i.to(self.device) for i in example]
            example = torch.cat(example, dim=0)
            example = example.unsqueeze(0)
            # print(example.shape, 'is the shape')
            # example = list(torch.cat((*orig, *example), dim=0))
        else:
            example = torch.from_numpy(example)
            example = example.to(self.device)
        if self.target_transform:
            label = self.target_transform(label)
        # print(len(example), "post_transforms")
        label = torch.from_numpy(label) # , dtype=torch.float32
        label = label.to(self.device)
        return example, label, labelf
    


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

