from .core import NamedTensor, contract, assert_match, lift
import numpy as np
import torch
from collections import OrderedDict
import torch.nn.functional as F
import pytest

def make_tensors(sizes):
    return [np.zeros(sizes), torch.zeros(sizes)]



def test_shift():
    for base in make_tensors([10, 2, 50]):
        ntensor = NamedTensor(base, 'alpha beta gamma')

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

        assert ntensor.shape() == OrderedDict([("beta", 2),
                                               ("gamma", 50),
                                               ("alpha", 10)])

def test_reduce():

    for base in make_tensors([10, 2, 50]):
        ntensor = NamedTensor(base, 'alpha beta gamma')

        ntensora = ntensor.reduce("alpha", "mean")
        assert ntensora.shape() == OrderedDict([("beta", 2),
                                               ("gamma", 50)])


        ntensorb = ntensor.reduce("alpha gamma", "sum")
        assert ntensorb.shape() == OrderedDict([("beta", 2)])


def test_apply():
    base = torch.zeros([10, 2, 50])
    ntensor = NamedTensor(base, 'alpha beta gamma')
    ntensor = ntensor.apply("alpha", F.softmax)
    assert pytest.approx(ntensor.reduce("alpha", "sum").tensor[0, 0].item(),
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
    ntensor3 = ntensor1.binop(torch.mul, ntensor2)
    assert base3.shape == ntensor3.tensor.shape
    assert (base3 == ntensor3.tensor).all()


def test_contract():
    base1 = np.random.randn(10, 2, 50)
    base2 = np.random.randn(10, 20, 2)
    ntensor1 = NamedTensor(base1, 'alpha beta gamma')
    ntensor2 = NamedTensor(base2, 'alpha delta beta')
    assert_match(ntensor1, ntensor2)

    base3 = np.einsum("abg,adb->a", base1, base2)

    ntensor3 = contract("alpha", ntensor1, ntensor2)
    assert ntensor3.shape() == OrderedDict([("alpha", 10)])
    assert ntensor3.tensor.shape == base3.shape
    print (ntensor3.tensor - base3)
    assert (np.abs(ntensor3.tensor - base3) < 1e-5).all()


        # ntensora = ntensor.reduce("alpha", "mean")
        # assert ntensora.shape() == OrderedDict([("beta", 2),
        #                                        ("gamma", 50)])


        # ntensorb = ntensor.reduce("alpha gamma", "mean")
        # assert ntensorb.shape() == OrderedDict([("beta", 2)])


def test_lift():
    def test_function(tensor):
        return np.sum(tensor, dim=1)

    base = np.random.randn(10, 70, 50)
    ntensor = NamedTensor(base, 'batch alpha beta')

    lifted = lift(test_function, ["alpha beta"], "beta")


    ntensor2 = lifted(ntensor)
    assert ntensor2.shape() == OrderedDict([("batch", 10),
                                            ("beta", 2)])
