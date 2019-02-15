from .. import ntorch
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
    ntorch.nn.Conv1d(5, 10, 2).spec("input", "time", "output")


def test_drop():
    drop = ntorch.nn.Dropout()
    n = ntorch.randn(4, 20, names=("batch", "target"))
    out = drop(n)
    assert n.shape == out.shape
    ntorch.nn.Conv1d(5, 10, 2).spec("input", "time", "output")


def test_cnn():
    conv = ntorch.nn.Conv1d(5, 10, 2).spec("input", "time", "output")
    n = ntorch.randn(20, 30, 5, names=("batch", "time", "input"))
    out = conv(n)
    print(out)
    assert out.shape == OrderedDict(
        [("batch", 20), ("output", 10), ("time", 29)]
    )


def test_rnn():
    rnn = ntorch.nn.RNN(5, 10, 3).spec("input", "time", "output")
    n = ntorch.randn(20, 30, 5, names=("batch", "time", "input"))
    out, state = rnn(n)
    assert out.shape == OrderedDict(
        [("batch", 20), ("time", 30), ("output", 10)]
    )
    print(state.shape)
    assert state.shape == OrderedDict(
        [("batch", 20), ("layers", 3), ("output", 10)]
    )

    out, state = rnn(n, state)
    assert out.shape == OrderedDict(
        [("batch", 20), ("time", 30), ("output", 10)]
    )
    print(state.shape)
    assert state.shape == OrderedDict(
        [("batch", 20), ("layers", 3), ("output", 10)]
    )


def test_lstm():
    rnn = ntorch.nn.LSTM(5, 10, 3).spec("input", "time", "output")
    n = ntorch.randn(20, 30, 5, names=("batch", "time", "input"))
    out, state = rnn(n)
    assert out.shape == OrderedDict(
        [("batch", 20), ("time", 30), ("output", 10)]
    )

    assert state[0].shape == OrderedDict(
        [("batch", 20), ("layers", 3), ("output", 10)]
    )

    out, state = rnn(n, state)
    assert out.shape == OrderedDict(
        [("batch", 20), ("time", 30), ("output", 10)]
    )
    assert state[0].shape == OrderedDict(
        [("batch", 20), ("layers", 3), ("output", 10)]
    )


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


def test_pad_1d():
    const_pad = ntorch.nn.ConstantPad1d((2, 0), 0).spec("cols", "rows")
    input = ntorch.tensor([[[1, 2, 3],
                            [1, 2, 3]]], names=["batch", "rows", "cols"])
    output = const_pad(input)
    print(output, output.shape)
    assert output.shape == OrderedDict([("batch", 1), ("cols", 3), ("rows", 4)])
    const_pad_2 = ntorch.nn.ConstantPad1d((2, 0), 0).spec("rows", "cols")
    output_2 = const_pad_2(input)
    print(output_2, output_2.shape)
    assert output_2.shape == OrderedDict([("batch", 1), ("rows", 2), ("cols", 5)])


def test_pad_2d():
    const_pad = ntorch.nn.ConstantPad2d((2, 0, 2, 0), 0).spec("batch", ("rows", "cols"))
    input = ntorch.tensor([[[1, 2, 3],
                            [1, 2, 3]]], names=["batch", "rows", "cols"])
    output = const_pad(input)
    print(output, output.shape)
    assert output.shape == OrderedDict([("batch", 1), ("rows", 4), ("cols", 5)])


def test_pad_3d():
    const_pad = ntorch.nn.ConstantPad3d((2, 0, 2, 0, 2, 0), 0).spec("firstdim", ("batch", "rows", "cols"))
    input = ntorch.tensor([[[[1, 2, 3],
                            [1, 2, 3]]]], names=["firstdim", "batch", "rows", "cols"])
    output = const_pad(input)
    print(output, output.shape)
    assert output.shape == OrderedDict([("firstdim", 1), ("batch", 3), ("rows", 4), ("cols", 5)])
