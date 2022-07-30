import copy
import sys
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as f

sys.path.append('../../e3nn/e3nn')
sys.path.append('../../e3nn')
from e3nn import Kernel
from e3nn.non_linearities.rescaled_act import swish
from e3nn.radial import CosineBasisModel
from e3nn.point.operations import Convolution
from e3nn.linear import Linear

import numpy as np

"""
Transformer without generator/decoder.
"""

PROT_ATOMS = ('C', 'O', 'N', 'S', 'P')

def make_model(Rs_in, Rs_hidden, num_heads, num_dense, encoder_layers, max_radius=10.0, 
                number_of_basis=3, h=100, L=1, act=swish, use_attention=True):
    c = copy.deepcopy
    attn = MultiHeadedAttention(Rs_hidden, num_heads, max_radius, number_of_basis, h, L, act)
    ff = PositionwiseFeedForward(Rs_hidden, num_dense, max_radius, number_of_basis, h, L, act)
    model = GertTransformer(
        Encoder(EncoderLayer(Rs_hidden, c(attn), c(ff), use_attention), Rs_hidden, encoder_layers),
        Embeddings(Rs_in, Rs_hidden, max_radius, number_of_basis, h, L, act))
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    num_parameters = sum([np.prod(p.size()) for p in model.parameters()])
    return model


class GertTransformer(nn.Module):
    def __init__(self, encoder, src_embed):
        """
        :param encoder: instance of class Encoder
        :param src_embed: instance of class Embeddings -- src_embed(arr) will run the forward fn
        :param generator: instance of class Generator
        """
        super(GertTransformer, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed

    def forward(self, xyz, elements, src_mask=None, attn_mask=None):
        "Take in and process masked src and target sequences."
        return self.encode(xyz, elements, src_mask=src_mask, attn_mask=attn_mask)

    def encode(self, xyz, elements, src_mask=None, attn_mask=None):
        x = self.encoder(self.src_embed(xyz, elements, mask=src_mask), xyz, mask=src_mask, attn_mask=attn_mask)
        return x


def clones(module, N):
    "Produces N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """
    Stack of N layers comprise the core encoder
    """

    def __init__(self, layer, Rs, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)

    def forward(self, features, xyz, mask=None, attn_mask=None):
        
        for layer in self.layers:
            features = layer(features, xyz, mask, attn_mask)

        return features


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    """

    def __init__(self, Rs, self_attn, feed_forward, use_attention):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = dimensionality(Rs)
        self.use_attention = use_attention

    def forward(self, features, xyz, mask, attn_mask):
        if self.use_attention:
            features = features + self.self_attn(features, xyz, mask=mask, attn_mask=attn_mask)
        features = features + self.feed_forward(features, xyz, mask=mask)

        return features


class MultiHeadedAttention(nn.Module):
    def __init__(self, Rs, heads, max_radius, number_of_basis, h, L, act):
        super(MultiHeadedAttention, self).__init__()
        self.conv = Convolution(Kernel(Rs, Rs, 
                               partial(CosineBasisModel, max_radius=max_radius, 
                               number_of_basis=number_of_basis, h=h, L=L, act=act)))
        self.query = Linear(Rs, Rs)
        self.eq_attn = clones(AttentionConvolution(Rs, max_radius, number_of_basis, h, L, act), heads)

    def forward(self, features, xyz, mask=None, attn_mask=None):
        features = torch.cat([attn(features, xyz, attn_mask=attn_mask).unsqueeze(-1) for attn in self.eq_attn], dim=-1)
        features = torch.sum(features, dim=-1) 
        if mask is not None:
            features = features * mask

        res = self.conv(features.float(), xyz.float())
        if mask is not None:
            res = res * mask

        return res


class AttentionConvolution(torch.nn.Module):
    def __init__(self, Rs, max_radius, number_of_basis, h, L, act):
        super().__init__()
        self.query = Linear(Rs, Rs)
        self.key = Kernel(Rs, Rs, 
                          partial(CosineBasisModel, max_radius=max_radius, 
                          number_of_basis=number_of_basis, h=h, L=L, act=act))
        self.value = Kernel(Rs, Rs, 
                            partial(CosineBasisModel, max_radius=max_radius, 
                            number_of_basis=3, h=100, L=1, act=swish))

    def forward(self, features, geometry, attn_mask=None, n_norm=1,
                custom_backward_kernel=False, r_eps=0):
        """
        :param features:     tensor [batch,  in_point, channel]
        :param geometry:     tensor [batch,  in_point, xyz]
        :param out_geometry: tensor [batch, out_point, xyz]
        :param n_norm: Divide kernel by sqrt(n_norm) before passing to convolution.
        :param custom_backward_kernel: call ConvolutionEinsumFn rather than using automatic differentiation
        :return:             tensor [batch, out_point, channel]
        """
        assert features.size()[:2] == geometry.size()[:2], "features size ({}) and geometry size ({}) should match".format(features.size(), geometry.size())
        rb = geometry.unsqueeze(1)  # [batch, 1, b, xyz]
        ra = geometry.unsqueeze(2)  # [batch, a, 1, xyz]

        k = self.key(rb.float() - ra.float(), custom_backward=custom_backward_kernel, r_eps=r_eps)  # [batch, a, b, i, j]
        k.div_(n_norm ** 0.5)
        v = self.value(rb.float() - ra.float(), custom_backward=custom_backward_kernel, r_eps=r_eps)  # [batch, a, b, i, j]
        v.div_(n_norm ** 0.5)

        query = self.query(features)
        key = torch.einsum("zabij,zbj->zabi", k, features)
        value = torch.einsum("zabij,zbj->zabi", v, features)

        d_k = query.size(-1)
        scores = torch.einsum("zabi,zai->zab", key, query) / d_k
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        p_attn = f.softmax(scores, dim=-1)
        features = torch.einsum("zab,zabi->zai", p_attn, value)
        return features


def dimensionality(Rs):
    return sum([channels * (2 * l + 1) for channels, l in Rs])


class PositionwiseFeedForward(nn.Module):
    def __init__(self, Rs, num_dense, max_radius, number_of_basis, h, L, act):
        """
        :param num_dense: number of dense layers after attention but before evaluation;
        """
        super(PositionwiseFeedForward, self).__init__()
        self.num_dense = num_dense
        self.linears = clones(Convolution(Kernel(Rs, Rs, 
                                  partial(CosineBasisModel, max_radius=max_radius, 
                                  number_of_basis=number_of_basis, h=h, L=L, act=act))), num_dense)

    def forward(self, features, xyz, mask=None):
        """
        Passes the prediction through the layers and returns the final layer's output
        """
        if self.num_dense == 0:
            return features

        for i in range(len(self.linears)):
            features = self.linears[i](features.float(), xyz.float())
            if mask is not None:
                features = features * mask 

        return features


class Embeddings(nn.Module):
    """
    Transforms input embeddings to the correct dimension d_model using a Linear layer.
    """

    def __init__(self, Rs_in, Rs_hidden, max_radius, number_of_basis, h, L, act):
        super(Embeddings, self).__init__()
        self.conv = Convolution(Kernel(Rs_in, Rs_hidden, partial(CosineBasisModel, 
                                max_radius=max_radius, number_of_basis=number_of_basis, 
                                h=h, L=L, act=act)))

    def forward(self, xyz, elements, mask=None):
        features = self.conv(elements.float(), xyz.float()) 
        
        if mask is not None:
            features = features * mask
        
        return features

