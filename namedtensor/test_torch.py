from . import NamedTensor, ntorch
import torch
import torch.nn.functional as F


def attention(query, key, value):
    return (
        key.contract("hidden", query).softmax("keys").contract("keys", value)
    )




def random_tensors(shape, num=1, requires_grad=False):
    tensors = [
        torch.randn(shape, requires_grad=requires_grad) for i in range(0, num)
    ]
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
        self.bM, self.br, self.w = random_tensors(
            [out_hid], num=3, requires_grad=True
        )

    def forward(self, y, ht, rt1):
        # -- [batch_size x hidden_dimension]
        tmp = torch.einsum("ik,kl->il", [ht, self.Wh]) + torch.einsum(
            "ik,kl->il", [rt1, self.Wr]
        )

        mt = torch.tanh(
            torch.einsum("ijk,kl->ijl", [y, self.WY])
            + tmp.unsqueeze(1).expand_as(y)
            + self.bM
        )
        # -- [batch_size x sequence_length]
        at = F.softmax(torch.einsum("ijk,k->ij", [mt, self.w]), dim=-1)

        # -- [batch_size x hidden_dimension]
        rt = torch.einsum("ijk,ij->ik", [y, at]) + torch.tanh(
            torch.einsum("ij,jk->ik", [rt1, self.Wt]) + self.br
        )

        # -- [batch_size x hidden_dimension], [batch_size x sequence_dimension]
        return rt, at


def random_ntensors(names, num=1, requires_grad=False):
    tensors = [
        ntorch.randn(names, requires_grad=requires_grad) for i in range(0, num)
    ]
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

    def forward(self, y, ht, rt1):
        tmp = ht.dot("inhid", self.Wh) + rt1.dot("inhid", self.Wr)
        at = (
            ntorch.tanh(y.dot("inhid", self.WY) + tmp + self.bM)
            .dot("outhid", self.w)
            .softmax("seqlen")
        )

        rt = y.dot("seqlen", at).stack(outhid=("inhid",)) + ntorch.tanh(
            rt1.dot("inhid", self.Wt) + self.br
        )

        return rt, at


def test_attention():
    # Sampled dummy inputs
    # -- [batch_size x sequence_length x hidden_dimension]
    y = torch.randn(3, 5, in_hid)
    # -- [batch_size x hidden_dimension]
    ht, rt1 = torch.randn(3, in_hid), torch.randn(3, in_hid)
    ea = EinsumAttention()
    r, a = ea.forward(y, ht, rt1)

    y = NamedTensor(y, ("batch", "seqlen", "inhid"))
    ht = NamedTensor(ht, ("batch", "inhid"))
    rt1 = NamedTensor(rt1, ("batch", "inhid"))
    nta = NamedTensorAttention()
    nr, na = nta.forward(y, ht, rt1)
