from . import assert_match, ntorch, NamedTensor
import numpy as np
import torch
from collections import OrderedDict
import pytest
import torch.nn.functional as F


def make_tensors(sizes, names):
    return [ntorch.randn(sizes, names=names)]


def test_shift():
    for ntensor in make_tensors((10, 2, 50), ("alpha", "beta", "gamma")):
        # Split
        ntensor = ntensor.split("alpha", ("delta", "epsilon"), delta=2)
        assert ntensor.vshape == (2, 5, 2, 50)

        # Merge
        ntensor = ntensor.stack(("delta", "epsilon"), "alpha")
        assert ntensor.vshape == (10, 2, 50)

        # Transpose
        ntensor = ntensor.transpose("beta", "alpha", "gamma")
        assert ntensor.vshape == (2, 10, 50)

        # Transpose
        ntensor = ntensor.rename("beta", "beta2")
        assert ntensor.shape == OrderedDict([("beta2", 2), ("alpha", 10), ("gamma", 50)])



def test_reduce():
    for ntensor in make_tensors((10, 2, 50), ("alpha", "beta", "gamma")):
        ntensora = ntensor.mean("alpha")
        assert ntensora.shape == OrderedDict([("beta", 2), ("gamma", 50)])

        ntensorb = ntensor.sum(("alpha", "gamma"))
        assert ntensorb.shape == OrderedDict([("beta", 2)])


def test_apply():
    base = torch.zeros([10, 2, 50])
    ntensor = ntorch.tensor(base, ("alpha", "beta", "gamma"))
    ntensor = ntensor.softmax("alpha")
    assert (ntorch.abs(ntensor.sum("alpha") - 1.0) < 1e-5).all()


def test_apply2():
    base = torch.zeros([10, 2, 50])
    ntensor = ntorch.tensor(base, ("alpha", "beta", "gamma"))
    ntensor = ntensor.op(F.softmax, dim="alpha")
    assert (ntorch.abs(ntensor.sum("alpha") - 1.0) < 1e-5).all()


def test_sum():
    base = torch.zeros([10, 2, 50])
    ntensor = ntorch.tensor(base, ("alpha", "beta", "gamma"))
    s = ntensor.sum()
    assert s.values == base.sum()


def test_fill():
    base = torch.zeros([10, 2, 50])
    ntensor = ntorch.tensor(base, ("alpha", "beta", "gamma"))
    ntensor.fill_(20)
    assert (ntensor == 20).all()


def test_mask():
    t = ntorch.tensor(torch.Tensor([[1, 2], [3, 4]]), ("a", "b"))
    mask = ntorch.tensor(torch.ByteTensor([[0, 1], [1, 0]]), ("a", "b"))
    ntensor = t.masked_select(mask, "c")
    assert ntensor.shape == OrderedDict([("c", 2)])


def test_gather():
    t = torch.Tensor([[1, 2], [3, 4]])
    base = torch.gather(t, 1, torch.LongTensor([[0, 0], [1, 0]]))

    t = ntorch.tensor(torch.Tensor([[1, 2], [3, 4]]), ("a", "b"))
    index = ntorch.tensor(torch.LongTensor([[0, 0], [1, 0]]), ("a", "c"))
    ntensor = ntorch.gather(t, index, c="b")
    assert (ntensor.values == base).all()

    x = ntorch.tensor(torch.rand(2, 5), ("c", "b"))
    y = ntorch.tensor(torch.rand(3, 5), ("a", "b"))
    y.scatter_(
        ntorch.tensor(
            torch.LongTensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), ("c", "b")
        ),
        x,
        a="c",
    )
    assert y.shape == OrderedDict([("a", 3), ("b", 5)])


def test_unbind():
    base = torch.zeros([10, 2, 50])
    ntensor = ntorch.tensor(base, ("alpha", "beta", "gamma"))
    out = ntensor.unbind("beta")
    assert len(out) == 2
    assert out[0].shape == OrderedDict([("alpha", 10), ("gamma", 50)])

    base = torch.zeros([10])
    ntensor = ntorch.tensor(base, ("alpha",))
    ntensor.fill_(20)
    c = ntensor.unbind("alpha")
    assert len(c) == 10
    assert c[0].item() == 20


@pytest.mark.xfail
def test_fail():
    for base1, base2 in zip(
        make_tensors([10, 2, 50]), make_tensors([10, 20, 2])
    ):
        ntensor1 = NamedTensor(base1, ("alpha", "beta", "gamma"))
        ntensor2 = NamedTensor(base2, ("alpha", "beat", "gamma"))
        assert_match(ntensor1, ntensor2)


def test_multiple():
    base1 = torch.rand([10, 2, 50])
    base2 = torch.rand([10, 20, 2])
    ntensor1 = ntorch.tensor(base1, ("alpha", "beta", "gamma"))
    ntensor2 = ntorch.tensor(base2, ("alpha", "delta", "beta"))
    assert_match(ntensor1, ntensor2)

    # Try applying a projected bin op
    base3 = torch.mul(base1.view([10, 1, 2, 50]), base2.view([10, 20, 2, 1]))
    ntensor3 = ntensor1.mul(ntensor2).transpose(
        "alpha", "delta", "beta", "gamma"
    )

    assert base3.shape == ntensor3.vshape
    assert (base3 == ntensor3.values).all()


def test_contract():
    base1 = torch.randn(10, 2, 50)
    ntensor1 = ntorch.tensor(base1, ("alpha", "beta", "gamma"))
    base2 = torch.randn(10, 20, 2)
    ntensor2 = ntorch.tensor(base2, ("alpha", "delta", "beta"))
    assert_match(ntensor1, ntensor2)

    base3 = torch.einsum("abg,adb->a", (base1, base2))

    ntensor3 = ntorch.dot(("beta", "gamma", "delta"), ntensor1, ntensor2)
    assert ntensor3.shape == OrderedDict([("alpha", 10)])
    assert ntensor3.vshape == base3.shape
    assert (np.abs(ntensor3._tensor - base3) < 1e-5).all()

    # ntensora = ntensor.reduce("alpha", "mean")
    # assert ntensora.named_shape == OrderedDict([("beta", 2),
    #                                        ("gamma", 50)])

    # ntensorb = ntensor.reduce("alpha gamma", "mean")
    # assert ntensorb.named_shape == OrderedDict([("beta", 2)])


# def test_lift():
#     def test_function(tensor):
#         return np.sum(tensor, dim=1)

#     base = np.random.randn(10, 70, 50)
#     ntensor = NamedTensor(base, 'batch alpha beta')

#     lifted = lift(test_function, ["alpha beta"], "beta")


#     ntensor2 = lifted(ntensor)
#     assert ntensor2.named_shape == OrderedDict([("batch", 10),
#                                             ("beta", 2)])


def test_unbind2():
    base1 = torch.randn(10, 2, 50)
    ntensor1 = ntorch.tensor(base1, ("alpha", "beta", "gamma"))
    a, b = ntensor1.unbind("beta")
    assert a.shape == OrderedDict([("alpha", 10), ("gamma", 50)])


def test_access():
    base1 = torch.randn(10, 2, 50)

    ntensor1 = ntorch.tensor(base1, ("alpha", "beta", "gamma"))

    assert (ntensor1.access("gamma")[45] == base1[:, :, 45]).all()
    assert (ntensor1.get("gamma", 1)._tensor == base1[:, :, 1]).all()

    assert (ntensor1.access("gamma beta")[45, 1] == base1[:, 1, 45]).all()


def test_takes():
    base1 = torch.randn(10, 2, 50)

    ntensor1 = ntorch.tensor(base1, ("alpha", "beta", "gamma"))
    indices = torch.ones(30).long()
    ntensor2 = ntorch.tensor(indices, ("indices",))

    selected = ntensor1.index_select("beta", ntensor2)
    assert (selected._tensor == base1.index_select(1, indices)).all()
    assert selected.shape == OrderedDict(
        [("alpha", 10), ("indices", 30), ("gamma", 50)]
    )


def test_narrow():
    base1 = torch.randn(10, 2, 50)

    ntensor1 = ntorch.tensor(base1, ("alpha", "beta", "gamma"))
    narrowed = ntensor1.narrow("gamma", 0, 25)
    assert narrowed.shape == OrderedDict(
        [("alpha", 10), ("beta", 2), ("gamma", 25)]
    )


# def test_ops():
#     base1 = ntorch.randn(dict(alpha=10, beta=2, gamma=50))
#     base2 = ntorch.log(base1)
#     base2 = ntorch.exp(base1)


@pytest.mark.xfail
def test_mask2():
    base1 = ntorch.randn(10, 2, 50, names=("alpha", "beta", "gamma"))
    base2 = base1.mask_to("alpha")
    print(base2._schema._masked)
    base2 = base2.softmax("alpha")


def test_unmask():
    base1 = ntorch.randn(10, 2, 50, names=("alpha", "beta", "gamma"))
    base2 = base1.mask_to("alpha")
    base2 = base2.mask_to("")
    base2 = base2.softmax("alpha")


def test_division():
    base1 = NamedTensor(torch.ones(3, 4), ("short", "long"))
    expected = NamedTensor(torch.ones(3) / 4, ("short",))
    assert_match(base1 / base1.sum("long"), expected)


def test_subtraction():
    base1 = ntorch.ones(3, 4, names=("short", "long"))
    base2 = ntorch.ones(3, 4, names=("short", "long"))
    expect = ntorch.zeros(3, 4, names=("short", "long"))
    assert_match(base1 - base2, expect)


def test_neg():
    base = ntorch.ones(3, names=("short",))
    expected = NamedTensor(-1 * torch.ones(3), ("short",))
    assert_match(-base, expected)


# def test_scalar():
#     base1 = ntorch.randn(dict(alpha=10, beta=2, gamma=50))
#     base2 = base1 + 10
