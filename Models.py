import numpy as np
import torch
import os
from torch import nn
from torch import optim
from einops import rearrange
from torch.nn import functional as F

#***********************************************
#Encoder and Discriminator has same architecture
#***********************************************
class Discriminator(nn.Module):
    def __init__(self, channel=512,out_class=1,is_dis =True):
        super(Discriminator, self).__init__()
        self.is_dis=is_dis
        self.channel = channel
        n_class = out_class 
        
        self.conv1 = nn.Conv3d(1, channel//8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(channel//4)
        self.conv3 = nn.Conv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(channel//2)
        self.conv4 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(channel)
        self.conv5 = nn.Conv3d(channel, n_class, kernel_size=4, stride=1, padding=0)
        
    def forward(self, x, _return_activations=False):
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        h5 = self.conv5(h4)
        output = h5
        
        return output
    
class Code_Discriminator(nn.Module):
    def __init__(self, code_size=100,num_units=750):
        super(Code_Discriminator, self).__init__()
        n_class = 1
        self.l1 = nn.Sequential(nn.Linear(code_size, num_units),
                                nn.BatchNorm1d(num_units),
                                nn.LeakyReLU(0.2,inplace=True))
        self.l2 = nn.Sequential(nn.Linear(num_units, num_units),
                                nn.BatchNorm1d(num_units),
                                nn.LeakyReLU(0.2,inplace=True))
        self.l3 = nn.Linear(num_units, 1)
        
    def forward(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        output = h3
            
        return output



class MultiHeadAttention(nn.Module):
    def __init__(self, k_dims=1000, heads=8, *args):
        super(MultiHeadAttention, self).__init__()
        self.k = k_dims
        self.heads = heads
        self.W_q = nn.Linear(self.k, self.k, bias=False)
        self.W_k = nn.Linear(self.k, self.k, bias=False)
        self.W_v = nn.Linear(self.k, self.k, bias=False)

        self.unify_heads = nn.Linear(self.k, self.k)

    def forward(self, x):
        b, t, k = x.shape
        h = self.heads
        s = k//h
        q = self.W_q(x)
        ky = self.W_k(x)
        v = self.W_v(x)
        q = q.view(b, t, h, s)
        ky = ky.view(b, t, h, s)
        v = v.view(b, t, h, s)
        # print('q', q.shape)
        # print('k', ky.shape)
        # print('v', v.shape)
        q = rearrange(q, "b t h s -> (b h) t s", t=1)
        ky = rearrange(ky, "b t h s -> (b h) t s", t=1)
        v = rearrange(v, "b t h s -> (b h) t s", t=1)
        # print('post_rearrangement')
        # print('q', q.shape)
        # print('k', ky.shape)
        # print('v', v.shape)
        w_hat = torch.bmm(q, ky.transpose(1,2))/(k**.5)
        # print('w_hat', w_hat.shape)
        w = torch.softmax(w_hat, dim=-1)
        # print('w', w.shape)
        # print("v", v.shape)
        # out = torch.matmul(w, v.transpose(1,2)).view(b,h,1,s)
        out = torch.bmm(w, v)
        # print('out', out.shape)
        out = rearrange(out, "(b h) t s -> b (h t s)", h=self.heads)
        out = self.unify_heads(out)
        # print('out', out.shape)
        return out


class AttentionM(nn.Module):
    def __init__(self, embed_d=1000, heads=8):
        super(AttentionM, self).__init__()
        self.embed_dim = embed_d
        self.heads = heads
        self.W_q = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.W_k = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.W_v = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.MhAtt = nn.MultiheadAttention(embed_d, heads, batch_first=True)
        
    def forward(self, x):
        b, t, k = x.shape
        h = self.heads
        s = k//h
        q = self.W_q(x)
        ky = self.W_k(x)
        v = self.W_v(x)
        # q = q.view(b, t, h, s)
        # ky = ky.view(b, t, h, s)
        # v = v.view(b, t, h, s)
        # print('q', q.shape)
        # print('k', ky.shape)
        # print('v', v.shape)
        # q = rearrange(q, "b t h s -> (b h) t s", t=1)
        # ky = rearrange(ky, "b t h s -> (b h) t s", t=1)
        # v = rearrange(v, "b t h s -> (b h) t s", t=1)
        out, wts = self.MhAtt(q, ky, v)
        return out


class Encoder(nn.Module):
    def __init__(self, channel=512,out_class=1000,is_dis =False):
        super(Encoder, self).__init__()
        self.is_dis=is_dis
        self.channel = channel
        n_class = out_class 
        
        self.conv1 = nn.Conv3d(1, channel//8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(channel//4)
        self.conv3 = nn.Conv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(channel//2)
        self.conv4 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(channel)
        self.conv5 = nn.Conv3d(channel, n_class, kernel_size=4, stride=1, padding=0)
        self.attention = MultiHeadAttention()
        
    def forward(self, x, _return_activations=False):
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        h5 = self.conv5(h4)
        h5 = rearrange(h5, 'b k n o p-> b n (k o p)', n=1,o=1,p=1)
        output = self.attention(h5)
        output = rearrange(output, 'b (k n o p) -> b k n o p', n=1,o=1,p=1)
        return output
    


class Generator(nn.Module):
    def __init__(self, noise:int=100, channel:int=64):
        super(Generator, self).__init__()
        _c = channel

        self.relu = nn.ReLU()
        self.noise = noise
        self.tp_conv1 = nn.ConvTranspose3d(noise, _c*8, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(_c*8)
        
        self.tp_conv2 = nn.Conv3d(_c*8, _c*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(_c*4)
        
        self.tp_conv3 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(_c*2)
        
        self.tp_conv4 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(_c)
        
        self.tp_conv5 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, noise):

        noise = noise.view(-1,self.noise,1,1,1)
        h = self.tp_conv1(noise)
        h = self.relu(self.bn1(h))
        
        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv2(h)
        h = self.relu(self.bn2(h))
     
        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv3(h)
        h = self.relu(self.bn3(h))

        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv4(h)
        h = self.relu(self.bn4(h))

        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv5(h)

        h = F.tanh(h)

        return h


class FeatureMapBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=7, padding=3, padding_mode='reflect')

    def forward(self, x):
        x = self.conv(x)
        return x
    

class ContractingBlock(nn.Module):
    def __init__(self, input_channels, use_bn=True, kernel_size=3, activation='relu'):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, input_channels * 2, kernel_size=kernel_size, padding=1, stride=2, padding_mode='reflect')
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_bn:
            self.instancenorm = nn.InstanceNorm3d(input_channels * 2)
        self.use_bn = use_bn

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x


class PatchGANdiscriminator(nn.Module):
    def __init__(self, in_C=1, hidden_c=64):
        super(PatchGANdiscriminator, self).__init__()
        self.upfeature = FeatureMapBlock(in_C, hidden_c)
        self.contract1 = ContractingBlock(hidden_c, use_bn=False, kernel_size=4, activation='lrelu')
        self.contract2 = ContractingBlock(hidden_c * 2, kernel_size=4, activation='lrelu')
        self.contract3 = ContractingBlock(hidden_c * 4, kernel_size=4, activation='lrelu')
        self.final = nn.Conv3d(hidden_c * 8, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        xn = self.final(x3)
        return xn