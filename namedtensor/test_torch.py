from . import NamedTensor, ntorch
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import pytest


def attention(query, key, value):
    return key.contract("hidden", query).softmax("keys").contract("keys", value)


def test_attention():
    batch = 50
    queries = 20
    keys = 15
    hidden = 10

    key_names = dict(batch=batch, key=keys, hidden=hidden)
    SimpleAttention().forward(
        ntorch.randn(dict(batch=batch, queries=queries, hidden=hidden)),
        ntorch.randn(key_names),
        ntorch.randn(key_names),
    )


# Parameters
# -- [hidden_dimension]


def random_tensors(shape, num=1, requires_grad=False):
    tensors = [torch.randn(shape, requires_grad=requires_grad) for i in range(0, num)]
    return tensors[0] if num == 1 else tensors


in_hid = 7
out_hid = 7

# -- [hidden_dimension x hidden_dimension]
weights = random_tensors([in_hid, out_hid], num=4, requires_grad=True)


class EinsumAttention:
    def __init__(self):

        torch.manual_seed(0)
        self.WY, self.Wh, self.Wr, self.Wt = random_tensors(
            [in_hid, out_hid], num=4, requires_grad=True
        )
        self.bM, self.br, self.w = random_tensors([out_hid], num=3, requires_grad=True)

    def forward(self, Y, ht, rt1):
        # -- [batch_size x hidden_dimension]
        tmp = torch.einsum("ik,kl->il", [ht, self.Wh]) + torch.einsum(
            "ik,kl->il", [rt1, self.Wr]
        )

        Mt = torch.tanh(
            torch.einsum("ijk,kl->ijl", [Y, self.WY])
            + tmp.unsqueeze(1).expand_as(Y)
            + self.bM
        )
        # -- [batch_size x sequence_length]
        at = F.softmax(torch.einsum("ijk,k->ij", [Mt, self.w]), dim=-1)

        # -- [batch_size x hidden_dimension]
        rt = torch.einsum("ijk,ij->ik", [Y, at]) + torch.tanh(
            torch.einsum("ij,jk->ik", [rt1, self.Wt]) + self.br
        )

        # -- [batch_size x hidden_dimension], [batch_size x sequence_dimension]
        return rt, at


def random_ntensors(names, num=1, requires_grad=False):
    tensors = [ntorch.randn(names, requires_grad=requires_grad) for i in range(0, num)]
    return tensors[0] if num == 1 else tensors


class NamedTensorAttention:
    def __init__(self):
        torch.manual_seed(0)
        self.WY, self.Wh, self.Wr, self.Wt = random_ntensors(
            dict(inhid=in_hid, outhid=out_hid), num=4, requires_grad=True
        )
        self.bM, self.br, self.w = random_ntensors(
            dict(outhid=out_hid), num=3, requires_grad=True
        )

    def forward(self, Y, ht, rt1):
        tmp = ht.contract("inhid", self.Wh) + rt1.contract("inhid", self.Wr)
        at = (
            ntorch.tanh(Y.contract("inhid", self.WY) + tmp + self.bM)
            .contract("outhid", self.w)
            .softmax("seqlen")
        )

        rt = Y.contract("seqlen", at).shift("inhid -> (outhid)") + ntorch.tanh(
            rt1.contract("inhid", self.Wt) + self.br
        )

        return rt, at


def test_attention():
    # Sampled dummy inputs
    # -- [batch_size x sequence_length x hidden_dimension]
    Y = torch.randn(3, 5, in_hid)
    # -- [batch_size x hidden_dimension]
    ht, rt1 = torch.randn(3, in_hid), torch.randn(3, in_hid)
    ea = EinsumAttention()
    r, a = ea.forward(Y, ht, rt1)

    Y = NamedTensor(Y, "batch seqlen inhid")
    ht = NamedTensor(ht, "batch inhid")
    rt1 = NamedTensor(rt1, "batch inhid")
    nta = NamedTensorAttention()
    nr, na = nta.forward(Y, ht, rt1)


def layer_norm(x):
    mean = x.reduce("mean", "hidden")
    std = x.reduce("std", "hidden")
    return (x - mean).contract("hidden", self.a_2) / (std + self.eps) + self.b_2
