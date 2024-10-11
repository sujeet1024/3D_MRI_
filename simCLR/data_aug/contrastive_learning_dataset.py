import os
import torch
import numpy as np
import torchio as tio
import nibabel as nib
from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from torch.utils.data import DataLoader, Dataset
from numpy.lib.stride_tricks import as_strided


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(64),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(64),
                                                              n_views),
                                                          download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
        




class BrainDataset(Dataset):
    def __init__(self, annotations_file, device, n_views, transform=None, target_transform=None):
        self.annotations = annotations_file #load csv file containing training examples using pandas
        self.transform = ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(64),
                                                                  n_views)
        self.target_transform = target_transform
        self.orig_transform = ContrastiveLearningViewGenerator(self.get_simclr_pipeline_trans_orig(64), n_views)
        self.device = device

    def __len__(self):
        return len(self.annotations)
    
    @staticmethod
    def get_simclr_pipeline_trans_orig(size, s=1):
        data_transforms = tio.Compose([])
        return data_transforms

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        # color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = tio.Compose([tio.RandomFlip(),
                                       tio.RandomSwap(4,4),
                                       tio.RandomBlur(),
                                       tio.RandomGamma()])
        return data_transforms

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
            example = np.swapaxes(example, 0, 2)
            example = np.flip(example, axis=2)
        example = normz(example, is_torch=False)
        example = np.expand_dims(example, axis=0)
        # print(example.shape, "pre_transforms")
        label = example.copy()
        if self.transform:
            orig = self.orig_transform(example)
            orig = [torch.from_numpy(i) for i in orig]
            orig = [i.to(self.device) for i in orig]
            example = self.transform(example)
            example = [torch.from_numpy(i) for i in example] # , dtype=torch.float32
            example = [i.to(self.device) for i in example]
            example = list(torch.cat((*orig, *example), dim=0))
        else:
            example = torch.from_numpy(example)
            example = example.to(self.device)
        if self.target_transform:
            label = self.target_transform(label)
        # print(len(example), "post_transforms")
        label = torch.from_numpy(label) # , dtype=torch.float32
        label = label.to(self.device)
        return example, label
    


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

