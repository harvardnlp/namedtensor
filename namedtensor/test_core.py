from .core import NamedTensor, contract, assert_match, lift, build
import numpy as np
import torch
from collections import OrderedDict
import torch.nn.functional as F
import pytest

def make_tensors(sizes):
    return [build(torch.randn, sizes)]


def test_shift():
    for ntensor in make_tensors(dict(alpha=10, beta=2, gamma=50)):

        # Split
        ntensor = ntensor.shift('alpha -> (delta epsilon)', delta = 2)
        assert ntensor.tensor.shape == (2, 5, 2, 50)

        # Merge
        ntensor = ntensor.shift('(delta epsilon) -> alpha')
        assert ntensor.tensor.shape == (10, 2, 50)

        # Transpose
        ntensor = ntensor.shift('beta alpha gamma')
        assert ntensor.tensor.shape == (2, 10, 50)

        # Promote
        ntensor = ntensor.shift('... alpha')
        assert ntensor.tensor.shape == (2, 50, 10)

        assert ntensor.named_shape == OrderedDict([("beta", 2),
                                                   ("gamma", 50),
                                                   ("alpha", 10)])

def test_reduce():

    for ntensor in make_tensors(dict(alpha=10, beta=2, gamma=50)):
        ntensora = ntensor.mean("alpha")
        assert ntensora.named_shape == OrderedDict([("beta", 2),
                                                    ("gamma", 50)])


        ntensorb = ntensor.sum("alpha gamma")
        assert ntensorb.named_shape == OrderedDict([("beta", 2)])


def test_apply():
    base = torch.zeros([10, 2, 50])
    ntensor = NamedTensor(base, 'alpha beta gamma')
    ntensor = ntensor.softmax("alpha")
    assert pytest.approx(ntensor.sum("alpha").tensor[0, 0].item(),
                         1.0)


@pytest.mark.xfail
def test_fail():
    for base1, base2 in zip(make_tensors([10, 2, 50]),
                            make_tensors([10, 20, 2])):
        ntensor1 = NamedTensor(base1, 'alpha beta gamma')
        ntensor2 = NamedTensor(base2, 'alpha beat gamma')
        assert_match(ntensor1, ntensor2)

def test_multiple():
    base1 = torch.rand([10, 2, 50])
    base2 = torch.rand([10, 20, 2])
    ntensor1 = NamedTensor(base1, 'alpha beta gamma')
    ntensor2 = NamedTensor(base2, 'alpha delta beta')
    assert_match(ntensor1, ntensor2)


    # Try applying a projected bin op
    base3 = torch.mul(base1.view([10, 1, 2, 50]),
                      base2.view([10, 20, 2, 1]))
    ntensor3 = ntensor1.mul(ntensor2).shift("alpha delta beta gamma")


    assert base3.shape == ntensor3.tensor.shape
    assert (base3 == ntensor3.tensor).all()


def test_contract():
    base1 = torch.randn(10, 2, 50)
    ntensor1 = NamedTensor(base1, 'alpha beta gamma')
    base2 = torch.randn(10, 20, 2)
    ntensor2 = NamedTensor(base2, 'alpha delta beta')
    assert_match(ntensor1, ntensor2)

    base3 = torch.einsum("abg,adb->a", base1, base2)

    ntensor3 = contract("beta gamma delta", ntensor1, ntensor2)
    assert ntensor3.named_shape == OrderedDict([("alpha", 10)])
    assert ntensor3.tensor.shape == base3.shape
    print (ntensor3.tensor - base3)
    assert (np.abs(ntensor3.tensor - base3) < 1e-5).all()


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

def test_unbind():
    base1 = torch.randn(10, 2, 50)
    ntensor1 = NamedTensor(base1, 'alpha beta gamma')
    a, b = ntensor1.unbind("beta")
    assert a.named_shape == OrderedDict([("alpha", 10),
                                         ("gamma", 50)])


def test_access():
    base1 = torch.randn(10, 2, 50)

    ntensor1 = NamedTensor(base1, 'alpha beta gamma')

    assert (ntensor1.access("gamma")[45] == base1[:, :, 45]).all()

    assert (ntensor1.access("gamma beta")[45, 1] == base1[:, 1, 45]).all()



def test_takes():
    base1 = torch.randn(10, 2, 50)

    ntensor1 = NamedTensor(base1, 'alpha beta gamma')
    indices = torch.ones(30).long()
    ntensor2 = NamedTensor(indices, "indices")

    selected = ntensor1.index_select("beta", ntensor2)
    assert (selected.tensor \
            == base1.index_select(1, indices)).all()
    print(selected.named_shape)
    assert selected.named_shape ==  \
        OrderedDict([("alpha", 10), ("indices", 30), ("gamma", 50)])

def test_narrow():
    base1 = torch.randn(10, 2, 50)

    ntensor1 = NamedTensor(base1, 'alpha beta gamma')
    narrowed = ntensor1.narrow("gamma -> ngamma", 0, 25)
    assert narrowed.named_shape ==  \
        OrderedDict([("alpha", 10), ("beta", 2), ("ngamma", 25)])
