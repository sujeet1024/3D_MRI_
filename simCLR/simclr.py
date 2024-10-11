import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models

import logging
import os
import sys
# import torch
import einops
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)



# r3d_18 = models.video.r3d_18(pretrained=True)
# # Change the first layer
# r3d_18.stem[0] = torch.nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)


class ResFeat(torch.nn.Module):
    def __init__(self, original_model):
        super(ResFeat, self).__init__()
        self.features = torch.nn.Sequential(*list(original_model.children())[:-2])
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512*4*4*4, 1000)
        self.proj = nn.Linear(1000, 128)

    
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.linear(x)
        proj = self.proj(x)
        return x, proj
    
# resfeat = ResFeat(r3d_18)

# x = torch.randn((8,1,64,64,64))
# repr, proj = resfeat(x)
# print(repr.shape)
# print(proj.shape)



class SimCLR(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(SimCLR, self).__init__()
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training2.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def forward(self, x):
        embed, proj = self.model(x)
        return embed, proj 
        
    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                # print(len(images))
                # for i in images:
                #     print(i.shape)
                images = torch.cat(images, dim=0)
                images = einops.rearrange(images, 'a b c d -> a 1 b c d')
                # print(images.shape)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    _,features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")






class Classifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim=64, out_class=2):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(.8),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_class)
        )

    def forward(self, x):
        return self.net(x)

