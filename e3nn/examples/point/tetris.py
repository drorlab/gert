# pylint: disable=C, R, not-callable, no-member, arguments-differ
from functools import partial
import torch
import torch.nn.functional as F
import sys
sys.path.append('../..')
from e3nn.networks import GatedConvNetwork
from e3nn.o3 import rand_rot
from e3nn import Kernel
from e3nn.non_linearities.rescaled_act import swish
from e3nn.radial import GaussianRadialModel, CosineBasisModel
from e3nn.point.operations import Convolution
from e3nn.linear import Linear
from e3nn.non_linearities.norm_activation import NormActivation
from e3nn.non_linearities.norm import Norm

sys.path.append('../../../gert2/atom3d_datasets')
from equivariant_transformer import make_model

def get_dataset():
    tetris = [[(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
              [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)],  # chiral_shape_2
              [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
              [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # T
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # zigzag
              [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]]  # L
    tetris = torch.tensor(tetris, dtype=torch.get_default_dtype())
    labels = torch.arange(len(tetris))

    # apply random rotation
    tetris = torch.einsum('ij,zaj->zai', rand_rot(), tetris)

    return tetris, labels


class Transformer(torch.nn.Module):
    def __init__(self, transformer, Rs_hidden, Rs_out):
        super().__init__()
        self.transformer = transformer
        self.fc1 = Convolution(Kernel(Rs_hidden, Rs_hidden, 
                        partial(GaussianRadialModel, max_radius=10.0, 
                        number_of_basis=3, h=100, L=1, act=swish)))
        self.norm = Convolution(Kernel(Rs_hidden, Rs_out, 
                        partial(GaussianRadialModel, max_radius=10.0, 
                        number_of_basis=3, h=100, L=1, act=swish)))
        self.fc2 = Linear(Rs_out, Rs_out)

    def forward(self, x, xyz):
        """
        x should be (batch_size, d_atom)
        """
        x = self.transformer(xyz.float(), x.float())
        x = self.fc1(x.float(), xyz.float())
        x = self.norm(x.float(), xyz.float())

        x = torch.sum(x, dim=1)
        mean = x.mean(-1, keepdims=True)
        std = x.std(-1, keepdims=True)
        eps = 1e-6
        x = (x - mean) / (std + eps)

        x = self.fc2(x.float())
        probs = F.softmax(x, dim=-1)
        return probs 

class SumNetwork(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.network = GatedConvNetwork(*args, **kwargs)

    def forward(self, *args, **kwargs):
        output = self.network(*args, **kwargs)
        return output.sum(-2)  # Sum over N


def main():
    torch.set_default_dtype(torch.float64)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tetris, labels = get_dataset()
    tetris = tetris.to(device)
    labels = labels.to(device)
    Rs_in = [(1, 0)]
    Rs_hidden = [(16, 0), (16, 1), (16, 2)]
    Rs_out = [(len(tetris), 0)]
    lmax = 3

    #f = SumNetwork(Rs_in, Rs_hidden, Rs_out, lmax)
    transformer = make_model(Rs_in, Rs_hidden, 1, 1, 1)
    f = Transformer(transformer, Rs_hidden, Rs_out)
    f = f.to(device)

    optimizer = torch.optim.Adam(f.parameters(), lr=1e-2)

    feature = tetris.new_ones(tetris.size(0), tetris.size(1), 1)

    for step in range(50):
        #print(feature.shape, tetris.shape)
        out = f(feature, tetris)
        loss = torch.nn.functional.cross_entropy(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = out.argmax(1).eq(labels).double().mean().item()
        print("step={} loss={} accuracy={}".format(step, loss.item(), acc))

    out = f(feature, tetris)

    r_tetris, _ = get_dataset()
    r_tetris = r_tetris.to(device)
    r_out = f(feature, r_tetris)

    print('equivariance error={}'.format((out - r_out).pow(2).mean().sqrt().item()))


if __name__ == '__main__':
    main()
