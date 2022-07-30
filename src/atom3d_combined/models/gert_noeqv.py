import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as f

sys.path.append('../../e3nn/e3nn')
sys.path.append('../../e3nn')
import rsh

from copy import deepcopy
import numpy as np


def make_model(
    num_heads=4, 
    d_model=8, 
    d_ff=16, 
    d_atom=4, 
    eta=0.4, 
    Rc=6.0, 
    num_atoms=2, 
    N=2, 
    num_dense=2, 
    use_attention=True,
    ):
    """
    :param input: inputted data via dataloader
    :param d_model: output of every sub-layer and embedding layer in the model; this facilitates residual connections
    (Layer(x + Sublayer(x)); big models will use d_model = 512
    :param d_atom: dimension of feature embedding
    :param n_total_possible_atoms: number of total distinct atom types in the possible space
    :param eta: used in radial factor of positional encoding
    :param Rc: radius within which neighbors are considered, in Angstrom; used in radial factor of positional encoding
    :return: the initialized model
    """
    attn = MultiHeadedAttention(
        num_heads, 
        d_model,
    )
    ff = PositionwiseFeedForward(
        d_model, 
        num_dense, 
        d_ff,
    )
    model = GertTransformer(
        Embeddings(
            AtomicPositionalEncoding(
                d_model, 
                num_atoms, 
                eta, 
                Rc,
            )
        ),
        Encoder(
            EncoderLayer(
                d_model, 
                deepcopy(attn), 
                deepcopy(ff),
                use_attention,
            ), 
            N,
        ),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    num_parameters = sum([np.prod(p.size()) for p in model.parameters()])
    return model


class GertTransformer(nn.Module):
    def __init__(self, embed, encoder):
        """
        :param encoder: instance of class Encoder
        :param src_embed: instance of class Embeddings
        """
        super(GertTransformer, self).__init__()
        self.embed = embed
        self.encoder = encoder

    def forward(self, neighbors, src_mask=None):
        embeddings = self.embed(neighbors)
        res = self.encoder(embeddings, src_mask)
        return res
        

class AtomicPositionalEncoding(nn.Module):
    def __init__(self, d_model, num_classes, eta, Rc):
        super(AtomicPositionalEncoding, self).__init__()
        assert d_model >= num_classes
        self.eta = eta
        self.Rc = Rc
        self.d_model = d_model
        self.num_classes = num_classes
        self.Nr = math.ceil(float(d_model) / num_classes) 
        self.lmax = math.floor(math.sqrt(float(self.Nr - 1)))
        self.L = torch.arange(self.lmax + 1).tolist()
        self.Rs = torch.arange(start=0, end=Rc, step=Rc/self.Nr)
        self.norm = LayerNorm(d_model) 
        if torch.cuda.is_available():
            self.Rs = self.Rs.cuda()
    
    def radial_term(self, r): 
        return torch.exp(-self.eta * (self.Rs[None, :] - r[..., None])**2)[..., :self.Nr]

    def angular_term(self, x):
        angular_term = rsh.spherical_harmonics_xyz(self.L, x)[..., :self.Nr]
        return angular_term

    def environment_weight(self, r):
        is_zero = r > self.Rc
        r[is_zero] = self.Rc
        return (0.5 * torch.cos(math.pi * r / self.Rc) + 0.5)[..., :self.Nr]
    
    def forward(self, x):
        batch_size, num_neighbors, _ = x.shape
        pos = torch.zeros(batch_size, num_neighbors, self.Nr, self.num_classes)
        r = torch.norm(x[..., :3], 2, dim=-1, keepdim=False)

        prod = self.angular_term(x[..., :3]) * self.radial_term(r.unsqueeze(-1)).squeeze() * self.environment_weight(r.unsqueeze(-1))

        if torch.cuda.is_available():
            x = x.to('cuda')
            pos = pos.to('cuda')
            r = r.to('cuda')
            prod = prod.to('cuda')

        for i in range(self.num_classes):
            pos[..., i][(x[..., 3] == float(i)) & (r != 0)] += prod[(x[..., 3] == float(i)) & (r != 0)]
        pos = pos.view(batch_size, num_neighbors, -1)[..., :self.d_model]
        pos[pos != pos] = 0
        pos = f.normalize(pos)
        mean = pos.mean(-1, keepdims=True)
        std = pos.std(-1, keepdims=True)
        eps = 1e-6
        res = (pos - mean) / (std + eps)
        return res


class Embeddings(nn.Module):
    def __init__(self, position):
        super(Embeddings, self).__init__()
        self.position = position

    def forward(self, x):
        return self.position(x)


class LayerNorm(nn.Module):
    """
    Construct a layernorm module
    Returns a normalized array, but along the features axis (not the samples)
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdims=True)
        std = x.std(-1, keepdims=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Encoder(nn.Module):
    """
    Stack of N layers comprise the core encoder
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    """
    def __init__(self, size, self_attn, feed_forward, use_attention):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_attention = use_attention
        self.norm = LayerNorm(self.size)

    def forward(self, x, mask):
        if self.use_attention:
            nx = self.norm(x)
            x = x + self.self_attn(nx, nx, nx, mask)
        res = x + self.feed_forward(self.norm(x))
        return res


def attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = f.softmax(scores, dim=-1)
    p_attn = p_attn.masked_fill(mask == 0, 0)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x.float()).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]         

        x, self.attn = attention(query, key, value, mask=mask)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        x = torch.squeeze(x)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, num_dense, d_ff, leaky_relu_slope=0.1):
        """
        :param num_dense: number of dense layers after attention but before evaluation;
        original transformer paper uses num_dense == 1
        """
        super(PositionwiseFeedForward, self).__init__()
        self.num_dense = num_dense
        if num_dense == 1:
            self.linears = clones(nn.Linear(d_model, d_model), 1)
        elif num_dense > 1:
            self.linears = nn.ModuleList()
            self.linears.append(nn.Linear(d_model, d_ff))
            self.linears.extend(clones(nn.Linear(d_ff, d_ff), num_dense - 2))
            self.linears.append(nn.Linear(d_ff, d_model))
        self.leaky_relu_slope = leaky_relu_slope
        self.dense_output_nonlinearity = lambda x: f.leaky_relu(x, negative_slope=self.leaky_relu_slope)

    def forward(self, x):
        """
        Passes the prediction through the layers and returns the final layer's output
        """
        if self.num_dense == 0:
            return x

        for i in range(len(self.linears) - 1):
            x = f.leaky_relu(self.linears[i](x.float()), negative_slope=self.leaky_relu_slope)

        res = self.dense_output_nonlinearity(self.linears[-1](x))
        return res


def clones(module, N):
    "Produces N identical layers."
    return nn.ModuleList([deepcopy(module) for _ in range(N)])