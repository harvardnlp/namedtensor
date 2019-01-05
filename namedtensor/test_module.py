from . import NamedTensor, ntorch
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import pytest


class WrappedModule(nn.Module):
    def __init__(self):
        super(WrappedModule, self).__init__()
        self.linear = nn.Linear(10, 20)

    def forward(self, inp):
        return inp.op(self.linear, shift="inhid -> outhid")


class NTModule(nn.Module):
    def __init__(self):
        super(NTModule, self).__init__()
        self.w = ntorch.randn(dict(inhid=10, outhid=20))
        self.w_param = nn.Parameter(self.w._tensor)

        self.b = ntorch.randn(dict(outhid=20))
        self.b_param = nn.Parameter(self.b._tensor)

    def forward(self, inp):
        return inp.contract("inhid", self.w) + self.b


def test_run():
    wm = WrappedModule()
    wm.forward(ntorch.randn(dict(batch=20, inhid=10)))
    nm = NTModule()
    nm.forward(ntorch.randn(dict(batch=20, inhid=10)))


def pe(d_model):
    pe = nt.zeros(dict(size=MAX_LEN, dmodel=d_model))
    position = NamedTensor(torch.arange(0, MAX_LEN).float(), "size")
    val = (
        NamedTensor(torch.arange(0, d_model, 2).float(), "tmp")
        .mul(-(math.log(10000.0) / d_model))
        .exp()
        .contract(position)
    )
    pe.access("dmodel")[0::2] = val.sin()
    pe.access("dmodel")[1::2] = val.cos()
    return pe


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim):
        super(Embedding, self).__init__()
        self.w = ntorch.randn(
            dict(numembeddings=num_embeddings, embeddingsdim=embeddings_dim)
        )
        self.w_param = nn.Parameter(self.w.tensor)

    def forward(self, inp):
        return self.w.index_select("numembeddings", inp)


def test_embedding():
    wm = Embedding(20, 50)
    out = wm.forward(ntorch.ones(dict(batch=20, seqlen=10)).long())
    print(out.named_shape)
    assert out.named_shape == OrderedDict(
        [("batch", 20), ("seqlen", 10), ("embeddingsdim", 50)]
    )
