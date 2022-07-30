import sys

from functools import partial
import torch
import torch.nn.functional as F
import torch.nn as nn

sys.path.append('../../e3nn/e3nn')
sys.path.append('../../e3nn')

from e3nn import Kernel
from e3nn.linear import Linear
from e3nn.non_linearities.norm import Norm
from e3nn.non_linearities.rescaled_act import swish
from e3nn.radial import GaussianRadialModel, CosineBasisModel
from e3nn.point.operations import Convolution


class EqvLBAFeedForward(torch.nn.Module):
    def __init__(self, Rs_hidden, Rs_ff, Rs_out, max_radius=10.0, number_of_basis=3, 
                 h=100, L=1, act=swish):
        super(EqvLBAFeedForward, self).__init__()
        self.fc1 = Convolution(Kernel(Rs_hidden, Rs_hidden, 
                               partial(GaussianRadialModel, max_radius=max_radius, 
                               number_of_basis=number_of_basis, h=h, L=L, act=act)))
        self.norm = Norm(Rs_hidden)
        self.fc2 = nn.Linear(dimensionality(Rs_ff), dimensionality(Rs_out))
        self.fc3 = nn.Linear(dimensionality(Rs_ff), dimensionality(Rs_ff))


    def forward(self, x, xyz, mask=None):
        x = self.fc1(x.float(), xyz.float())
        x = self.norm(x)

        if mask is not None:
            x[mask == 0] = 0

        x = torch.sum(x, dim=1)
        x = normalize(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.fc2(x.float())
        return x.view(-1)


class LBAFeedForward(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(LBAFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)


    def forward(self, x, mask=None):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = normalize(torch.mean(x, dim=1))
        x = self.fc2(x.float())
        return x.view(-1)
    

class EqvLEPFeedForward(torch.nn.Module):
    def __init__(self, Rs_hidden, Rs_ff, Rs_out, max_radius, number_of_basis, h, L):
        super(EqvLEPFeedForward, self).__init__()
        self.fc1 = Convolution(Kernel(Rs_hidden, Rs_hidden, partial(GaussianRadialModel,
                                        max_radius=max_radius, number_of_basis=number_of_basis, 
                                        h=h, L=L, act=swish)))
        self.norm = Norm(Rs_hidden)
        self.fc2 = Linear(Rs_ff, Rs_out)
        self.fc3 = Linear(Rs_ff, Rs_ff)

    def forward(self, input1, input2, xyz1, xyz2, mask=None):
        x = torch.cat((input1, input2), dim=1)
        xyz = torch.cat((xyz1, xyz2), dim=1)
        x = self.fc1(x.float(), xyz.float())
        x = self.norm(x)
        if mask is not None:
            x[mask == 0] = 0
        x = normalize(torch.sum(x, dim=1))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc2(x.float())
        return torch.sigmoid(x).view(-1)


class LEPFeedForward(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(LEPFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, input1, input2):
        x = torch.cat((input1, input2), dim=1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = normalize(torch.mean(x, dim=1))
        x = self.fc3(x.float())
        return torch.sigmoid(x).view(-1)


class EqvMSPFeedForward(torch.nn.Module):
    def __init__(self, Rs_hidden, Rs_ff, Rs_out, max_radius, number_of_basis, h, L):
        super(EqvMSPFeedForward, self).__init__()
        self.fc1 = Convolution(Kernel(Rs_hidden, Rs_hidden, 
                               partial(CosineBasisModel, max_radius=max_radius, 
                               number_of_basis=number_of_basis, h=h, L=L, act=swish)))
        self.norm = Norm(Rs_hidden)
        self.fc2 = Linear(Rs_ff, Rs_out)
        self.fc3 = Linear(Rs_ff, Rs_ff)
        self.i = 0


    def forward(self, input1, input2, xyz1, xyz2, mask=None):
        x = torch.cat((input1, input2), dim=1)
        xyz = torch.cat((xyz1, xyz2), dim=1)
        x = self.fc1(x.float(), xyz.float())
        x = self.norm(x)

        if mask is not None:
            x[mask == 0] = 0

        x = torch.sum(x, dim=1)

        mean = x.mean(-1, keepdims=True)
        std = x.std(-1, keepdims=True)
        eps = 1e-6
        x = (x - mean) / (std + eps)

        x = F.leaky_relu(self.fc3(x))
        x = self.fc2(x.float())
        self.i = self.i + 1

        return torch.sigmoid(x).view(-1)


class MSPFeedForward(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(MSPFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, input1, input2):
        x = torch.cat((input1, input2), dim=1)
        x = F.relu(self.fc1(x.float()))
        x = F.dropout(x, p=0.1)
        x = self.fc2(x)
        x = torch.mean(x, dim=1)
        return torch.sigmoid(x).view(-1)


class EqvPPIFeedForward(torch.nn.Module):
    def __init__(self, Rs_hidden, Rs_ff, Rs_out, max_radius, number_of_basis, h, L):
        super(EqvPPIFeedForward, self).__init__()
        self.fc1 = Convolution(Kernel(Rs_hidden, Rs_hidden, 
                               partial(CosineBasisModel, max_radius=max_radius, 
                               number_of_basis=number_of_basis, h=h, L=L, act=swish)))
        self.norm = Norm(Rs_hidden)
        self.fc2 = Linear(Rs_ff, Rs_out)
        self.fc3 = Linear(Rs_ff, Rs_ff)
        self.i = 0


    def forward(self, input1, input2, xyz1, xyz2, mask=None, normal_siamese=False, mask1=None, mask2=None, true_eqv=False):
        if true_eqv:
            output1 = self.fc1(input1.float(), xyz1.float())
            output1 = self.norm(output1)
            if mask1 is not None: 
                diag = torch.diagonal(mask, dim1=-2, dim2=-1)
                output1[diag == 0] = 0
            output2 = self.fc1(input2.float(), xyz2.float())
            output2 = self.norm(output2)
            if mask2 is not None: 
                diag = torch.diagonal(mask2, dim1=-2, dim2=-1)
                output2[diag == 0] = 0
            x = torch.cat((output1, output2), dim=1)
            x = normalize(torch.sum(x, dim=1))
        elif normal_siamese:
            output1 = self.fc1(input1.float(), xyz1.float())
            output1 = self.norm(output1, xyz1.float())
            if mask1 is not None: 
                diag = torch.diagonal(mask, dim1=-2, dim2=-1)
                output1[diag == 0] = 0
            output1 = normalize(torch.sum(output1, dim=1))

            output2 = self.fc1(input2.float(), xyz2.float())
            output2 = self.norm(output2, xyz2.float())
            if mask2 is not None: 
                diag = torch.diagonal(mask2, dim1=-2, dim2=-1)
                output2[diag == 0] = 0
            output2 = normalize(torch.sum(output2, dim=1))
            x = torch.cat((output1, output2), dim=-1)
        else:
            self.i = self.i + 1
            x = torch.cat((input1, input2), dim=1)
            xyz = torch.cat((xyz1, xyz2), dim=1)
            x = self.fc1(x.float(), xyz.float())
            x = self.norm(x)

            if mask is not None:
                x[mask == 0] = 0

            x = normalize(torch.sum(x, dim=1))

        x = F.leaky_relu(self.fc3(x))
        x = self.fc2(x.float())
        return torch.sigmoid(x).view(-1)


class PPIFeedForward(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(PPIFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, input1, input2):
        x = torch.cat((input1, input2), dim=1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = normalize(torch.mean(x, dim=1))
        x = self.fc3(x.float())
        return torch.sigmoid(x).view(-1)


class EqvPSRFeedForward(torch.nn.Module):
    def __init__(self, Rs_hidden, Rs_ff, Rs_out, max_radius, number_of_basis, h, L):
        super(EqvPSRFeedForward, self).__init__()
        self.fc1 = Convolution(Kernel(Rs_hidden, Rs_hidden, 
                               partial(GaussianRadialModel, max_radius=max_radius, 
                               number_of_basis=number_of_basis, h=h, L=L, act=swish)))
        self.norm = Norm(Rs_hidden)
        self.fc2 = Linear(Rs_ff, Rs_out)
        self.fc3 = Linear(Rs_ff, Rs_ff)


    def forward(self, x, xyz, mask=None):
        x = self.fc1(x.float(), xyz.float())
        x = self.norm(x)

        if mask is not None:
            diag = torch.diagonal(mask, dim1=-2, dim2=-1)
            x[diag == 0] = 0

        x = torch.sum(x, dim=1)
        x = normalize(x)

        x = F.leaky_relu(self.fc3(x))
        x = self.fc2(x.float())
        return torch.sigmoid(x).view(-1)


class PSRFeedForward(nn.Module):
    def __init__(self, d_model, n_out):
        super(PSRFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, n_out)
        self.fc3 = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        x = F.relu(self.fc3(x))

        if mask is not None:
            diag = torch.diagonal(mask, dim1=-2, dim2=-1)
            x[diag == 0] = 0    

        x = torch.sum(x, dim=1) 
        x = normalize(x)

        x = self.fc1(x.float())
        x = F.relu(x)
        x = self.fc2(x)

        return x.view(-1)


class EqvRESFeedForward(nn.Module):
    def __init__(self, Rs_hidden, Rs_out, max_radius, number_of_basis, h, L):
        super(EqvRESFeedForward, self).__init__()
        self.fc1 = Convolution(Kernel(Rs_hidden, Rs_hidden, 
                        partial(GaussianRadialModel, max_radius=max_radius, 
                        number_of_basis=number_of_basis, h=h, L=L, act=swish)))
        self.norm = Convolution(Kernel(Rs_hidden, Rs_out, 
                        partial(GaussianRadialModel, max_radius=max_radius, 
                        number_of_basis=number_of_basis, h=h, L=L, act=swish)))
        self.fc2 = Linear(Rs_out, Rs_out)


    def forward(self, x, xyz, mask=None):
        if mask is not None:
            diag = torch.diagonal(mask, dim1=-2, dim2=-1)
            x[diag == 0] = 0

        x = self.fc1(x.float(), xyz.float())
        x = self.norm(x.float(), xyz.float())
        if mask is not None:
            diag = torch.diagonal(mask, dim1=-2, dim2=-1)
            x[diag == 0] = 0

        x = torch.sum(x, dim=1)
        x = normalize(x)
        x = self.fc2(x.float())
        probs = F.softmax(x, dim=-1)
        return probs
    

class RESFeedForward(nn.Module):
    def __init__(self, d_model, n_out):
        super(RESFeedForward, self).__init__()
        self.proj = nn.Linear(d_model, n_out)

    def forward(self, x):
        x = torch.sum(x, dim=1)
        probs = F.softmax(self.proj(x.float()), dim=-1)
        return probs


class EqvRSRFeedForward(torch.nn.Module):
    def __init__(self, Rs_hidden, Rs_ff, Rs_out, max_radius, number_of_basis, h, L):
        super(EqvRSRFeedForward, self).__init__()
        self.fc1 = Convolution(Kernel(Rs_hidden, Rs_hidden, 
                               partial(GaussianRadialModel, max_radius=max_radius, 
                               number_of_basis=number_of_basis, h=h, L=L, act=swish)))
        self.norm = Norm(Rs_hidden)
        self.fc2 = Linear(Rs_ff, Rs_out)
        self.fc3 = Linear(Rs_ff, Rs_ff)


    def forward(self, x, xyz, mask=None):
        x = self.fc1(x.float(), xyz.float())
        x = self.norm(x)

        if mask is not None:
            diag = torch.diagonal(mask, dim1=-2, dim2=-1)
            x[diag == 0] = 0

        x = torch.sum(x, dim=1)
        x = normalize(x)

        x = F.leaky_relu(self.fc3(x))
        x = self.fc2(x.float())
        return x.view(-1)


class RSRFeedForward(nn.Module):
    def __init__(self, d_model, n_out):
        super(RSRFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, n_out)
        self.fc3 = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        x = F.relu(self.fc3(x))

        if mask is not None:
            diag = torch.diagonal(mask, dim1=-2, dim2=-1)
            x[diag == 0] = 0    

        x = torch.sum(x, dim=1) 
        x = normalize(x)

        x = self.fc1(x.float())
        x = F.leaky_relu(x)
        x = self.fc2(x)

        return x.view(-1)


def dimensionality(Rs):
    return sum([channels * (2 * l + 1) for channels, l in Rs])


def normalize(x, eps=1e-6):
    mean = x.mean(-1, keepdims=True)
    std = x.std(-1, keepdims=True)
    x = (x - mean) / (std + eps)
    return x