from . import ntorch
import torch.nn
from collections import OrderedDict

def test_nn():
    lin = ntorch.nn.Linear(20, 10).rename(output="input")
    out = lin(ntorch.randn(dict(batch=5, input=20)))
    assert out.shape == OrderedDict([("batch", 5), ("output", 10)])


def test_loss():
    loss = ntorch.nn.NLLLoss().reduce(["batch", "target"])

    predict = ntorch.randn(dict(batch=4, target=20))
    target = ntorch.tensor([2, 2, 3, 4], ["batch"])
    out = loss(predict, target)
    assert out.shape == OrderedDict([])

def test_drop():
    drop = ntorch.nn.Dropout()
    n = ntorch.randn(dict(batch=4, target=20))
    out = drop(n)
    assert n.shape == out.shape


class MyModule(ntorch.nn.Module):
    def __init__(self, in_dim, out_dim):
        self.in_name = in_dim[0]
        self.out_name = out_dim[0]

        super(MyModule, self).__init__()
        weight = ntorch.randn(dict((in_dim, out_dim)))
        self.register_parameter("weight", weight)
        bias = ntorch.randn(dict((out_dim,)))
        self.register_parameter("bias", bias)


    def forward(self, input):
        return self.weight.dot(self.in_name, input) + self.bias

def test_nn2():
    lin = MyModule(("input", 20) , ("output", 10))
    lin(ntorch.randn(dict(input=20)))



def test_embedding():
    embedding = ntorch.nn.Embedding(20, 10).augment("extra")

    input = ntorch.tensor([1, 2, 3], ("start",))

    out = embedding(input)
    assert out.shape == OrderedDict([("start", 3), ("extra", 10)])
