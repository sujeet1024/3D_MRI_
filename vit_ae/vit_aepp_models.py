from functools import partial
import os

import torch
from torchvision import models as tv

from collections import namedtuple
from torch import nn
import torch.nn.functional as F
import math

import numpy as np

# from vit_ae_env import PROJECT_ROOT_DIR
# from ..vit_ae_env import PROJECT_ROOT_DIR
PROJECT_ROOT_DIR  = os.path.abspath('../')
# PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def make_gaussian_kernel(sigma):
    ks = int(sigma * 5)
    if ks % 2 == 0:
        ks += 1
    ts = torch.linspace(-ks // 2, ks // 2 + 1, ks)
    gauss = torch.exp((-(ts / sigma) ** 2 / 2))
    kernel = gauss / gauss.sum()

    return kernel


def perform_3d_gaussian_blur(original_vol, blur_sigma=2):
    _, in_ch, _, _, _ = original_vol.shape
    return_vol = []
    for idx in range(in_ch):
        input_vol = original_vol[:, idx:idx+1]
        k = make_gaussian_kernel(blur_sigma)
        k3d = torch.einsum('i,j,k->ijk', k, k, k).to(input_vol.device)
        k3d = k3d / k3d.sum()
        vol_3d = F.conv3d(input_vol, k3d.reshape(1, 1, *k3d.shape), stride=1, padding=len(k) // 2)
        return_vol.append(vol_3d)
    return torch.cat(return_vol, dim=1)


class vgg_perceptual_loss(torch.nn.Module):
    def __init__(self, requires_grad=False, use_imagenet=False):
        super(vgg_perceptual_loss, self).__init__()

        if use_imagenet:
            print("Using VGG-Imagenet pretrained model for perceptual loss")
            vgg_pretrained_features = tv.vgg16(pretrained=True).eval().features
        else:
            print("Using VGG SSL features for training")
            vgg_pretrained_model = tv.vgg16(pretrained=False)
            model_path = os.path.join(PROJECT_ROOT_DIR, 'model', 'ckp-399.pth')
            vgg_pretrained_model.load_state_dict(
                torch.load(model_path), strict=False)
            vgg_pretrained_features = vgg_pretrained_model.eval().features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        self.N_slices = 4
        self.mse_loss = torch.nn.MSELoss()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward_one_view(self, X):
        # X = bs, ch, z, y, x
        X = X.permute(0, 2, 1, 3, 4)  # bs, z, ch, y, x
        X = X.reshape(-1, *X.size()[2:])
        if X.size(1) == 1:
            X = X.repeat(1, 3, 1, 1)
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)

        return out

    def compute_loss_per_channel(self, X1, X2):
        view1_activations = self.forward_one_view(X1)
        view2_activations = self.forward_one_view(X2)
        loss = torch.mean(
            torch.as_tensor([self.mse_loss(view1_activations[i], view2_activations[i]) for i in range(self.N_slices)]))
        return loss

    def forward(self, X1, X2):
        bs, ch, _, _, _ = X1.shape
        loss = 0
        for idx in range(ch):
            loss += self.compute_loss_per_channel(X1[:, idx:idx+1], X2[:, idx:idx+1])
        return loss / ch

    # def forward(self, X1, X2):
    #     view1_activations = self.forward_one_view(X1)
    #     view2_activations = self.forward_one_view(X2)
    #     loss = torch.mean(
    #         torch.as_tensor([self.mse_loss(view1_activations[i], view2_activations[i]) for i in range(self.N_slices)]))
    #     return loss


class SobelFilter3d(nn.Module):
    def __init__(self):
        super(SobelFilter3d, self).__init__()
        self.sobel_filter = self.setup_filter()

    def setup_filter(self):
        sobel_filter = nn.Conv3d(1, 3, kernel_size=3, stride=1, padding=1)
        sobel_filter.weight.data[0, 0].copy_(
            torch.FloatTensor(
                [
                    [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                    [[2, 0, -2], [4, 0, -4], [2, 0, -2]],
                    [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
                ]))
        sobel_filter.weight.data[1, 0].copy_(
            torch.FloatTensor([
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            ]))
        sobel_filter.weight.data[2, 0].copy_(
            torch.FloatTensor([
                [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
            ])
        )
        sobel_filter.bias.data.zero_()
        for p in sobel_filter.parameters():
            p.requires_grad = False
        return sobel_filter

    def forward(self, x):
        bs, ch, l, h, w = x.shape
        combined_edge_map = 0
        for idx in range(ch):
            g_x = self.sobel_filter(x[:, idx:idx+1])[:, 0]
            g_y = self.sobel_filter(x[:, idx:idx+1])[:, 1]
            g_z = self.sobel_filter(x[:, idx:idx+1])[:, 2]
            combined_edge_map += torch.sqrt((g_x **2 + g_y ** 2 + g_z ** 2))
        return combined_edge_map


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_l = np.arange(grid_size, dtype=np.float32)
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_l, grid_h, grid_w)  # Different from the github impl. Look at https://github.com/facebookresearch/mae/issues/18
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([-1, 1, grid_size, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use one third of dimensions to encode grid_l, grid_h and final dim for grid_w
    res = embed_dim // 3
    if res % 2 != 0:
        res += 1
    factor_w = embed_dim - 2 * res
    emb_l = get_1d_sincos_pos_embed_from_grid(res, grid[0])  # (L*H*W, D//3)
    emb_h = get_1d_sincos_pos_embed_from_grid(res, grid[1])  # (L*H*W, D//3)
    emb_w = get_1d_sincos_pos_embed_from_grid(factor_w, grid[2])  # (L*H*W, D-D//3-D//3)

    emb = np.concatenate([emb_l, emb_h, emb_w], axis=1)  # (L*H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)  # Default step size is 1 [1,2,3...]
    omega /= embed_dim / 2.  # Now value = [1/(d/2), 2/(d/2), 3/(d/2)...] = [1*2/d, 2* 2/d, 3*2/d, ...]
    omega = 1. / 10000 ** omega  # (D/2,) [1/10000^(1* 2/d), 1/10000^(2*2/d), 1/10000^(3*2/d), ...]
    # 1/10000^(2*k/d) and this is the definition of pe_j for the vector
    pos = pos.reshape(-1)  # (M,)
    # Different elements in the sequence start with different frequencies like the Furior series. Hence,
    # the multiplication with M, which is essentially a list of positions [0, 1, 2, ...] produces the
    # required effect of different start frequencies and linear increase of this start frequencies in different
    # columns.
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def traid(t):
    return t if isinstance(t, tuple) else (t, t, t)


# We have to adapt the different layers for 3D and try to align this with the `timm` implementations
class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, volume_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        volume_size = traid(volume_size)
        patch_size = traid(patch_size)
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.grid_size = (volume_size[0] // patch_size[0], volume_size[1] // patch_size[1], volume_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L, H, W = x.shape
        assert L == self.volume_size[0] and H == self.volume_size[1] and W == self.volume_size[2], \
            f"Volume image size ({L}*{H}*{W}) doesn't match model ({self.volume_size[0]}*{self.volume_size[1]}*{self.volume_size[2]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Mlp3D(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# The Attention is always applied to the sequences. Thus, at this point, it should be the same model
# whether we apply it in NLP, ViT, speech or any other domain :)
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp3D(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x




class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, volume_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, args=None):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed3D(volume_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        self.embed_dim = embed_dim
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 3 * in_chans, bias=True)  # encoder to decoder
        self.sobel_filter3D = SobelFilter3d()
        self.perceptual_loss = vgg_perceptual_loss(use_imagenet=True)   # args.use_imagenet)
        self.args = args
        self.perceptual_weight = 1 if self.args is None else self.args.perceptual_weight
        print(f"Using perceptual weight of {self.perceptual_weight}")

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], round(self.patch_embed.num_patches ** (1 / 3)),
                                            # int(self.patch_embed.num_patches ** (1/3)),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
        #                                             int(self.patch_embed.num_patches ** .5), cls_token=True)
        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    round(self.patch_embed.num_patches ** (1 / 3)), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, volume):
        """
        volume: (N, 3, L, H, W)
        x: (N, L, patch_size**3 *3)
        """
        p = self.patch_embed.patch_size[0]  # Patch size
        assert volume.shape[2] == volume.shape[3] == volume.shape[4] and volume.shape[
            2] % p == 0  # Ensuring we have the same dimension

        l = h = w = volume.shape[2] // p  # Since volumes have the same dimension. Possible limitation??
        x = volume.reshape(shape=(volume.shape[0], -1, l, p, h, p, w, p))
        x = torch.einsum('nclrhpwq->nlhwrpqc', x)
        x = x.reshape(shape=(volume.shape[0], l * h * w, -1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**3 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        l = h = w = round(x.shape[1] ** (1 / 3))
        assert l * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], l, h, w, p, p, p,
                             -1))  # Earlier 3 was hard-coded here. Maybe this way, we are more flexible with the number of channels
        x = torch.einsum('nlhwrpqc->nclrhpwq', x)
        volume = x.reshape(shape=(x.shape[0], -1, h * p, h * p, h * p))
        return volume

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        # print('rmshape:', L, 'mr')
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask, edge_map_weight=0):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = self.get_weighted_loss(pred, target, mask, edge_map_weight=edge_map_weight)
        return loss

    def get_weighted_loss(self, pred, target, mask, edge_map_weight=0):
        pred_vol, target_vol = self.unpatchify(pred), self.unpatchify(target)
        pred_edge_map, orig_input_edge_map = self.sobel_filter3D(pred_vol), self.sobel_filter3D(
            perform_3d_gaussian_blur(target_vol, blur_sigma=2))
        raw_edge_map_loss = F.mse_loss(pred_edge_map, orig_input_edge_map, reduction="mean")
        edge_map_loss = edge_map_weight * F.mse_loss(pred_edge_map, orig_input_edge_map, reduction="mean")
        reconstruction_loss = ((pred - target) ** 2).mean(dim=-1)
        reconstruction_loss = (reconstruction_loss * mask).sum() / mask.sum()  # mean loss on removed patches
        # Including the perceptual loss
        with torch.no_grad():
            percep_loss = self.perceptual_weight * self.perceptual_loss(pred_vol, target_vol)
        loss = edge_map_loss + reconstruction_loss + percep_loss
        return [loss, raw_edge_map_loss, reconstruction_loss, percep_loss]

    def forward(self, sample, mask_ratio=0.75, edge_map_weight=0):
        latent, mask, ids_restore = self.forward_encoder(sample, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(sample, pred, mask, edge_map_weight=edge_map_weight)
        return loss, pred, mask


class ContrastiveMAEViT(MaskedAutoencoderViT):
    def __init__(self, volume_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, args=None, use_proj=False):
        super().__init__(volume_size=volume_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                         depth=depth,
                         num_heads=num_heads, decoder_num_heads=decoder_num_heads, decoder_embed_dim=decoder_embed_dim,
                         decoder_depth=decoder_depth, mlp_ratio=mlp_ratio, norm_layer=norm_layer,
                         norm_pix_loss=norm_pix_loss, args=args)

        # build a 3-layer projector
        self.use_proj = use_proj
        if use_proj:
            self.projection_head = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim, bias=False),
                                                 nn.BatchNorm1d(self.embed_dim),
                                                 nn.ReLU(inplace=True),  # first layer
                                                 nn.Linear(self.embed_dim, self.embed_dim, bias=False),
                                                 nn.BatchNorm1d(self.embed_dim),
                                                 nn.ReLU(inplace=True),  # second layer
                                                 nn.Linear(self.embed_dim, self.embed_dim, bias=False),
                                                 nn.BatchNorm1d(self.embed_dim, affine=False))  # output layer
        self.predictor = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(self.embed_dim, self.embed_dim)  # output layer
        )

    def forward(self, view1, view2, mask_ratio=0.75, edge_map_weight=0):
        # First call the forward method of the autoencoder
        latent1, mask, ids_restore = super(ContrastiveMAEViT, self).forward_encoder(view1, mask_ratio)
        pred = super(ContrastiveMAEViT, self).forward_decoder(latent1, ids_restore)  # [N, L, p*p*3]
        loss = super(ContrastiveMAEViT, self).forward_loss(view1, pred, mask, edge_map_weight=edge_map_weight)

        # Call the encoder for the second view
        latent2, _, _ = self.forward_encoder(view2, mask_ratio)
        # Now, we perform the contrastive part on the embeddings
        # First we reshape the tensors so that they can be handled easily
        latent1 = latent1.view(-1, latent1.shape[2])
        latent2 = latent2.view(-1, latent2.shape[2])

        p1 = self.predictor(latent1)
        p2 = self.predictor(latent2)
        return loss, pred, mask, p1, p2, latent1.detach(), latent2.detach()


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def contr_mae_vit_base_patch16_dec512d8b(**kwargs):
    model = ContrastiveMAEViT(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
contr_mae_vit_base_patch16 = contr_mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks

if __name__ == '__main__':
    image_size = (64, 64, 64)
    model = contr_mae_vit_base_patch16(volume_size=image_size, in_chans=1, patch_size=16)
    sample_img = torch.randn(8, 1, 64, 64, 64)
    loss, pred, mask, p1, p2, z1, z2 = model(sample_img, sample_img)  # for contrastive
    # loss, pred, mask = model(sample_img, .75)
    pred = model.unpatchify(pred)
    print(pred.shape)
    print(loss[0].item())
    print(loss[1].item())
    print(loss[2].item())
    print(loss[3].item())

