# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member, redefined-builtin
from functools import partial

import torch

from e3nn import o3, rs
from e3nn.linear_mod import KernelLinear
from e3nn.util.sparse import get_sparse_buffer, register_sparse_buffer


class LearnableTensorSquare(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, linear=True, allow_change_output=False, allow_zero_outputs=False):
        super().__init__()

        self.Rs_in = rs.simplify(Rs_in)
        self.Rs_out = rs.simplify(Rs_out)

        ls = [l for _, l, _ in self.Rs_out]
        selection_rule = partial(o3.selection_rule, lfilter=lambda l: l in ls)

        if linear:
            Rs_in = [(1, 0, 1)] + self.Rs_in
        else:
            Rs_in = self.Rs_in
        self.linear = linear

        Rs_ts, T = rs.tensor_square(Rs_in, selection_rule)
        register_sparse_buffer(self, 'T', T)  # [out, in1 * in2]

        ls = [l for _, l, _ in Rs_ts]
        if allow_change_output:
            self.Rs_out = [(mul, l, p) for mul, l, p in self.Rs_out if l in ls]
        elif not allow_zero_outputs:
            assert all(l in ls for _, l, _ in self.Rs_out)

        self.kernel = KernelLinear(Rs_ts, self.Rs_out)  # [out, in, w]

    def __repr__(self):
        return "{name} ({Rs_in} -> {Rs_out})".format(
            name=self.__class__.__name__,
            Rs_in=rs.format_Rs(self.Rs_in),
            Rs_out=rs.format_Rs(self.Rs_out),
        )

    def forward(self, features):
        '''
        :param features: [..., channels]
        '''
        *size, n = features.size()
        features = features.reshape(-1, n)
        assert n == rs.dim(self.Rs_in)

        if self.linear:
            features = torch.cat([features.new_ones(features.shape[0], 1), features], dim=1)
            n += 1

        T = get_sparse_buffer(self, 'T')  # [out, in1 * in2]
        kernel = (T.t() @ self.kernel().T).T.reshape(rs.dim(self.Rs_out), n, n)  # [out, in1, in2]
        features = torch.einsum('zi,zj->zij', features, features)
        features = torch.einsum('kij,zij->zk', kernel, features)
        return features.reshape(*size, -1)


class LearnableTensorProduct(torch.nn.Module):
    def __init__(self, Rs_in1, Rs_in2, Rs_out, allow_change_output=False):
        super().__init__()

        self.Rs_in1 = rs.simplify(Rs_in1)
        self.Rs_in2 = rs.simplify(Rs_in2)
        self.Rs_out = rs.simplify(Rs_out)

        ls = [l for _, l, _ in self.Rs_out]
        selection_rule = partial(o3.selection_rule, lfilter=lambda l: l in ls)

        Rs_ts, T = rs.tensor_product(self.Rs_in1, self.Rs_in2, selection_rule)
        register_sparse_buffer(self, 'T', T)  # [out, in1 * in2]

        ls = [l for _, l, _ in Rs_ts]
        if allow_change_output:
            self.Rs_out = [(mul, l, p) for mul, l, p in self.Rs_out if l in ls]
        else:
            assert all(l in ls for _, l, _ in self.Rs_out)

        self.kernel = KernelLinear(Rs_ts, self.Rs_out)  # [out, in, w]

    def forward(self, features_1, features_2):
        """
        :return:         tensor [..., channel]
        """
        *size, n = features_1.size()
        features_1 = features_1.reshape(-1, n)
        assert n == rs.dim(self.Rs_in1)
        *size2, n = features_2.size()
        features_2 = features_2.reshape(-1, n)
        assert size == size2

        T = get_sparse_buffer(self, 'T')  # [out, in1 * in2]
        kernel = (T.t() @ self.kernel().T).T.reshape(rs.dim(self.Rs_out), rs.dim(self.Rs_in1), rs.dim(self.Rs_in2))  # [out, in1, in2]
        features = torch.einsum('kij,zi,zj->zk', kernel, features_1, features_2)
        return features.reshape(*size, -1)
