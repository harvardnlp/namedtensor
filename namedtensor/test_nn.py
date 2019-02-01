from . import ntorch
from collections import OrderedDict


def test_nn():
    lin = ntorch.nn.Linear(20, 10).spec("input", "output")
    out = lin(ntorch.randn(20, 5, names=("input", "batch")))
    assert out.shape == OrderedDict([("batch", 5), ("output", 10)])


def test_loss():
    loss = ntorch.nn.NLLLoss().spec("target")

    predict = ntorch.randn(20, 4, names=("target", "batch"))
    target = ntorch.tensor([2, 2, 3, 4], ["batch"])
    out = loss(predict, target)
    assert out.shape == OrderedDict([])


def test_drop():
    drop = ntorch.nn.Dropout()
    n = ntorch.randn(4, 20, names=("batch", "target"))
    out = drop(n)
    assert n.shape == out.shape


class MyModule(ntorch.nn.Module):
    def __init__(self, in_dim, out_dim, names):
        self.in_name, self.out_name = names

        super(MyModule, self).__init__()
        weight = ntorch.randn(in_dim, out_dim, names=names)
        self.register_parameter("weight", weight)
        bias = ntorch.randn(out_dim, names=(self.out_name,))
        self.register_parameter("bias", bias)

    def forward(self, input):
        return self.weight.dot(self.in_name, input) + self.bias


def test_nn2():
    lin = MyModule(20, 10, names=("input", "output"))
    lin(ntorch.randn(20, names=("input",)))


def test_embedding():
    embedding = ntorch.nn.Embedding(20, 10).spec("start", "extra")

    input = ntorch.tensor([1, 2, 3], ("start",))

    out = embedding(input)
    print(out.shape)
    assert out.shape == OrderedDict([("start", 3), ("extra", 10)])
