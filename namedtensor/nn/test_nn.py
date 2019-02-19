from .. import ntorch
from collections import OrderedDict
from hypothesis import given
from ..strategies import named_tensor, dim, dims, name
from hypothesis.strategies import data, integers, permutations
from hypothesis.extra.numpy import array_shapes

## HYPOTHESIS Tests
@given(data(), named_tensor())
def test_hyp_nn(data, x):
    d = data.draw(dim(x))
    n = data.draw(name(x))
    i = data.draw(integers(min_value=1, max_value=10))
    lin = ntorch.nn.Linear(x.shape[d], i).spec(d, n)
    out = lin(x)
    assert out.shape[n] == i
    assert d not in out.shape


@given(data(), named_tensor(shape=array_shapes(2, 6, min_side=2, max_side=5)))
def test_hyp_conv1(data, x):
    d, time = data.draw(dims(x, max_size=2))
    n = data.draw(name(x))
    i = data.draw(integers(min_value=1, max_value=10))
    conv = ntorch.nn.Conv1d(x.shape[d], i, 2).spec(d, time, n)
    out = conv(x)
    assert out.shape[n] == i
    assert d not in out.shape

    # All others stay the same
    for d_batch in x.dims:
        if d_batch not in [d, time]:
            assert out.shape[d_batch] == x.shape[d_batch]

    out = ntorch.nn.MaxPool1d(2).spec(time)(x)
    assert set(x.dims) == set(out.dims)


@given(data(), named_tensor(shape=array_shapes(3, 6, min_side=2, max_side=5)))
def test_hyp_conv2(data, x):
    d, time1, time2 = data.draw(dims(x, min_size=3, max_size=3))
    n = data.draw(name(x))
    i = data.draw(integers(min_value=1, max_value=10))
    conv = ntorch.nn.Conv2d(x.shape[d], i, (2, 2)).spec(d, (time1, time2), n)
    out = conv(x)
    assert out.shape[n] == i
    assert d not in out.shape

    # All others stay the same
    for d_batch in x.dims:
        if d_batch not in [d, time1, time2]:
            assert out.shape[d_batch] == x.shape[d_batch]

    out = ntorch.nn.MaxPool2d((2, 2)).spec((time1, time2))(x)
    assert set(x.dims) == set(out.dims)


@given(data(), named_tensor(shape=array_shapes(4, 6, min_side=2, max_side=5)))
def test_hyp_conv3(data, x):
    d, time1, time2, time3 = data.draw(dims(x, min_size=4, max_size=4))
    n = data.draw(name(x))
    i = data.draw(integers(min_value=1, max_value=10))
    conv = ntorch.nn.Conv3d(x.shape[d], i, (2, 2, 2)).spec(
        d, (time1, time2, time3), n
    )
    out = conv(x)
    assert out.shape[n] == i
    assert d not in out.shape

    # All others stay the same
    for d_batch in x.dims:
        if d_batch not in [d, time1, time2, time3]:
            assert out.shape[d_batch] == x.shape[d_batch]

    out = ntorch.nn.MaxPool3d((2, 2, 2)).spec((time1, time2, time3))(x)
    assert set(x.dims) == set(out.dims)


@given(data(), named_tensor(shape=array_shapes(2, 5, min_side=2, max_side=5)))
def test_hyp_loss(data, x):
    for loss_cls in [ntorch.nn.NLLLoss, ntorch.nn.CrossEntropyLoss]:
        ds = data.draw(dims(x, max_size=3))
        d = ds[0]
        max_class = x.shape[d]
        matches = [d2 for d2 in x.dims if d2 != d]
        target = ntorch.randint(
            max_class, [x.shape[d2] for d2 in matches], names=matches
        )
        target = target.transpose(*data.draw(permutations(target.dims)))
        print(target.shape, x.shape, d)

        loss = loss_cls().spec(d)
        out = loss(x, target)
        assert len(out.dims) == 0

        loss = loss_cls(reduction="none").spec(d)
        out2 = loss(x, target)
        assert len(out2.dims) == len(target.dims)


## OLD Tests
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


def test_pad():
    # Test 1d
    const_pad = ntorch.nn.ConstantPad1d((2, 0), 0).spec("rows")
    input = ntorch.tensor(
        [[[1, 2, 3], [1, 2, 3]]], names=["batch", "rows", "cols"]
    )
    output = const_pad(input)
    print(output, output.shape)
    assert output.shape == OrderedDict(
        [("batch", 1), ("cols", 3), ("rows", 4)]
    )
    const_pad = ntorch.nn.ConstantPad1d((2, 0), 0).spec("cols")
    output = const_pad(input)
    print(output, output.shape)
    assert output.shape == OrderedDict(
        [("batch", 1), ("rows", 2), ("cols", 5)]
    )

    # Test 2d
    const_pad_2d = ntorch.nn.ConstantPad2d((2, 0, 2, 0), 0).spec(
        ("rows", "cols")
    )
    output = const_pad_2d(input)
    print(output, output.shape)
    assert output.shape == OrderedDict(
        [("batch", 1), ("rows", 4), ("cols", 5)]
    )

    # Test 3d
    const_pad_3d = ntorch.nn.ConstantPad3d((2, 0, 2, 0, 2, 0), 0).spec(
        ("batch", "rows", "cols")
    )
    output = const_pad_3d(input)
    print(output, output.shape)
    assert output.shape == OrderedDict(
        [("batch", 3), ("rows", 4), ("cols", 5)]
    )
