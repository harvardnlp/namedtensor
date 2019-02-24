from . import assert_match, ntorch
import torch
from collections import OrderedDict
import pytest
import torch.nn.functional as F
from hypothesis import given
from .strategies import (
    named_tensor,
    broadcast_named_tensor,
    mask_named_tensor,
    dim,
    dims,
    name,
    names,
)
from hypothesis.strategies import (
    sampled_from,
    lists,
    data,
    floats,
    integers,
    permutations,
)


## HYPOTHESIS Tests
@given(data(), named_tensor())
def test_stack_basic(data, x):
    s = data.draw(dims(x))
    n = data.draw(name(x))
    x = x.stack(list(s), n)
    assert n in x.dims
    assert not (x.shape.keys() & s)


@given(named_tensor())
def test_deepcopy(x):
    import copy

    x2 = copy.deepcopy(x)
    assert id(x2.values) != id(x.values)
    assert torch.equal(x2.values, x.values)
    assert id(x2._schema) != id(x._schema)
    assert x2._schema._names == x._schema._names
    assert x2._schema._masked == x._schema._masked


@given(data(), named_tensor())
def test_unique(data, x):
    s = data.draw(dim(x))
    nu, ni = data.draw(names(x, max_size=2))
    output, inverse_indices = ntorch.unique(
        x, sorted=True, return_inverse=True, dim=s, names=(nu, ni)
    )
    assert all(
        output[{nu: i}] <= output[{nu: i + 1}]
        for i in range(output.shape[nu] - 1)
    )
    assert torch.equal(
        x.values, output.index_select(nu, inverse_indices).values
    )

    output, inverse_indices = ntorch.unique(
        x, sorted=True, return_inverse=True, names=(nu, ni)
    )
    assert all(
        (output[{nu: i}] <= output[{nu: i + 1}]).values
        for i in range(output.shape[nu] - 1)
    )
    assert torch.equal(
        x.values, output.index_select(nu, inverse_indices).values
    )


@given(data(), named_tensor())
def test_rename(data, x):
    s = data.draw(dim(x))
    n = data.draw(name(x))
    x = x.rename(s, n)
    assert n in x.dims
    assert s not in x.dims


@given(data(), named_tensor())
def test_split(data, x):
    s = data.draw(dim(x))
    ns = list(data.draw(names(x)))
    x2 = x.split(s, ns, **{n: 1 for n in ns[:-1]})
    assert len(set(ns) & set(x2.dims)) == len(ns)
    assert s not in x2.dims
    assert torch.prod(torch.tensor([x2.shape[n] for n in ns])) == x.shape[s]


@given(data(), named_tensor())
def test_reduce(data, x):
    ns = data.draw(dims(x))
    method = data.draw(sampled_from(sorted(x._reduce)))
    if method in ["squeeze"]:
        return

    if method not in ["logsumexp"]:
        y = getattr(x, method)()
        print(y)
        # assert y.values == getattr(x.values, method)()

    x2 = getattr(x, method)(tuple(ns))
    assert set(x2.dims) | set(ns) == set(x.dims)


@given(data(), named_tensor())
def test_binary_op(data, x):
    y = data.draw(broadcast_named_tensor(x))
    method = data.draw(sampled_from(sorted(x._binop)))
    x2 = getattr(x, method)(y)
    assert set(x2.dims) == set(x.dims) | set(y.dims)
    x3 = getattr(y, method)(x)
    assert set(x3.dims) == set(x.dims) | set(y.dims)


@given(data(), named_tensor())
def test_noshift(data, x):
    method = data.draw(
        sampled_from(sorted(x._noshift)).filter(lambda a: a not in {"cuda"})
    )
    x2 = getattr(x, method)()
    assert set(x2.dims) == set(x.dims)


@given(data(), named_tensor())
def test_apply(data, x):
    method = data.draw(
        sampled_from(sorted(x._noshift_dim | x._noshift_nn_dim))
    )
    s = data.draw(dim(x))
    x2 = getattr(x, method)(s)
    assert x.shape == x2.shape


@given(named_tensor())
def test_sum(x):
    s = x.sum()
    print(x.shape)
    assert s.values == x.values.sum()


@given(data(), named_tensor())
def test_mask(data, x):
    mask = data.draw(mask_named_tensor(x))
    x2 = x.masked_select(mask, "c")
    x2 = x[mask]
    print(x2)


@pytest.mark.xfail
@given(data(), named_tensor())
def test_maskfail(data, x):
    mask = data.draw(broadcast_named_tensor(x))
    x2 = x.masked_select(mask, "c")
    x2 = x[mask]
    print(x2)


@given(data(), named_tensor(), floats(allow_nan=False, allow_infinity=False))
def test_all_scalar_ops(data, x, y):
    x = x + y
    x = x - y
    x = x * y
    x = x / y

    x = y + x
    x = y - x
    x = y * x

    x = -x


def test_mod():
    base1 = ntorch.tensor(torch.Tensor([[1, 2, 3], [3, 4, 5]]), ("a", "b"))
    expected = ntorch.tensor(torch.Tensor([[1, 2, 0], [0, 1, 2]]), ("a", "b"))
    assert_match(ntorch.fmod(base1, 3), expected)


@given(data(), named_tensor())
def test_indexing(data, x):
    d = data.draw(dim(x))
    i = data.draw(integers(min_value=0, max_value=x.shape[d] - 1))
    x2 = x[{d: i}]
    assert set(x2.dims) == set(x.dims) - set([d])

    ds = data.draw(dims(x))
    index = {}
    for d in ds:
        i = data.draw(integers(min_value=0, max_value=x.shape[d] - 1))
        index[d] = i
    x2 = x[index]
    assert set(x2.dims) == set(x.dims) - set(ds)

    ds = data.draw(dims(x))
    index = {}
    for d in ds:
        i = data.draw(integers(min_value=0, max_value=x.shape[d] - 1))
        j = data.draw(integers(min_value=i + 1, max_value=x.shape[d]))
        index[d] = slice(i, j)
    x2 = x[index]
    assert set(x2.dims) == set(x.dims)
    x[index] = 6


@given(data(), named_tensor())
def test_tensor_indexing(data, x):
    d = data.draw(dim(x))
    indices = data.draw(
        lists(integers(min_value=0, max_value=x.shape[d] - 1), unique=True)
    )
    n = data.draw(name(x))
    ind_vector = ntorch.tensor(indices, names=n).long()
    x2 = x[{d: ind_vector}]
    assert set(x2.dims) == (set(x.dims) | set([n])) - set([d])

    x[{d: ind_vector}] = 5
    assert (x[{d: ind_vector}] == 5).all()


@given(data(), named_tensor())
def test_tensor_mask(data, x):
    mask = data.draw(mask_named_tensor(x))
    x[mask] = 6
    x2 = x[mask]
    assert x2.dims == ("on",)


@given(data(), named_tensor())
def test_cat(data, x):
    perm = data.draw(permutations(x.dims))
    y = x.transpose(*perm)
    for s in set(x.dims) & set(y.dims):
        c = ntorch.cat([x, y], dim=s)
        c = ntorch.cat([x, c], dim=s)
        c = ntorch.cat([c, x, y], dim=s)
    print(c)


@given(data(), named_tensor())
def test_stack(data, x):
    perm = data.draw(permutations(x.dims))
    print(perm)
    y = x.transpose(*perm)
    n = data.draw(name(x))
    z = ntorch.stack([x, y], n)
    assert set(z.dims) == set(x.dims) | set([n])


@given(data(), named_tensor())
def test_dot(data, x):
    y = data.draw(broadcast_named_tensor(x))
    dsx = data.draw(dims(x))
    dsy = data.draw(dims(x))
    x.dot(dsx, y)
    x.dot(dsy, y)
    y.dot(dsx, x)
    y.dot(dsy, x)


## Old style tests


def test_apply2():
    base = torch.zeros([10, 2, 50])
    ntensor = ntorch.tensor(base, ("alpha", "beta", "gamma"))
    ntensor = ntensor.op(F.softmax, dim="alpha")
    assert (ntorch.abs(ntensor.sum("alpha") - 1.0) < 1e-5).all()


# def test_fill():
#     base = torch.zeros([10, 2, 50])
#     ntensor = ntorch.tensor(base, ("alpha", "beta", "gamma"))
#     ntensor.fill_(20)
#     assert (ntensor == 20).all()


def test_gather():
    t = torch.Tensor([[1, 2], [3, 4]])
    base = torch.gather(t, 1, torch.LongTensor([[0, 0], [1, 0]]))

    t = ntorch.tensor(torch.Tensor([[1, 2], [3, 4]]), ("a", "b"))
    index = ntorch.tensor(torch.LongTensor([[0, 0], [1, 0]]), ("a", "c"))
    ntensor = ntorch.gather(t, "b", index, "c")
    assert (ntensor.values == base).all()
    assert ntensor.shape == OrderedDict([("a", 2), ("c", 2)])

    x = ntorch.tensor(torch.rand(2, 5), ("c", "b"))
    y = ntorch.tensor(torch.rand(3, 5), ("a", "b"))
    y.scatter_(
        "a",
        ntorch.tensor(
            torch.LongTensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), ("c", "b")
        ),
        x,
        "c",
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


# @pytest.mark.xfail
# def test_fail():
#     for base1, base2 in zip(
#         make_tensors([10, 2, 50]), make_tensors([10, 20, 2])
#     ):
#         ntensor1 = NamedTensor(base1, ("alpha", "beta", "gamma"))
#         ntensor2 = NamedTensor(base2, ("alpha", "beat", "gamma"))
#         assert_match(ntensor1, ntensor2)


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


# def test_contract():
#     base1 = torch.randn(10, 2, 50)
#     ntensor1 = ntorch.tensor(base1, ("alpha", "beta", "gamma"))
#     base2 = torch.randn(10, 20, 2)
#     ntensor2 = ntorch.tensor(base2, ("alpha", "delta", "beta"))
#     assert_match(ntensor1, ntensor2)

#     base3 = torch.einsum("abg,adb->a", (base1, base2))

#     ntensor3 = ntorch.dot(("beta", "gamma", "delta"), ntensor1, ntensor2)
#     assert ntensor3.shape == OrderedDict([("alpha", 10)])
#     assert ntensor3.vshape == base3.shape
#     assert (np.abs(ntensor3._tensor - base3) < 1e-5).all()

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


# def test_access():
#     base1 = torch.randn(10, 2, 50)

#     ntensor1 = ntorch.tensor(base1, ("alpha", "beta", "gamma"))

#     assert (ntensor1.access("gamma")[45] == base1[:, :, 45]).all()
#     assert (ntensor1.get("gamma", 1)._tensor == base1[:, :, 1]).all()

#     assert (ntensor1.access("gamma beta")[45, 1] == base1[:, 1, 45]).all()


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


# def test_division():
#     base1 = NamedTensor(torch.ones(3, 4), ("short", "long"))
#     expected = NamedTensor(torch.ones(3) / 4, ("short",))
#     assert_match(base1 / base1.sum("long"), expected)


# def test_scalarmult():
#     base1 = NamedTensor(torch.ones(3, 4), ("short", "long"))
#     rmul = 3 * base1
#     lmul = base1 * 3
#     assert_match(rmul, lmul)


# def test_subtraction():
#     base1 = ntorch.ones(3, 4, names=("short", "long"))
#     base2 = ntorch.ones(3, 4, names=("short", "long"))
#     expect = ntorch.zeros(3, 4, names=("short", "long"))
#     assert_match(base1 - base2, expect)


# def test_rightsubtraction():
#     base1 = ntorch.ones(3, 4, names=("short", "long"))
#     expect = ntorch.zeros(3, 4, names=("short", "long"))
#     assert_match(1 - base1, expect)


# def test_rightaddition():
#     base1 = ntorch.ones(3, 4, names=("short", "long"))
#     expect = NamedTensor(2 * torch.ones(3, 4), names=("short", "long"))
#     assert_match(1 + base1, expect)


# def test_neg():
#     base = ntorch.ones(3, names=("short",))
#     expected = NamedTensor(-1 * torch.ones(3), ("short",))
#     assert_match(-base, expected)


def test_nonzero():

    # only zeros
    x = ntorch.zeros(10, names=("alpha",))
    y = x.nonzero()
    assert x.shape == OrderedDict([("alpha", 10)])
    assert y.shape == OrderedDict([("elements", 0), ("inputdims", 1)])

    # `names` length must be 2
    y = x.nonzero(names=("a", "b"))
    assert y.shape == OrderedDict([("a", 0), ("b", 1)])

    # 1d tensor
    x = ntorch.tensor([0, 1, 2, 0, 5], names=("dim",))
    y = x.nonzero()
    assert 3 == y.size("elements")
    assert x.shape == OrderedDict([("dim", 5)])
    assert y.shape == OrderedDict([("elements", 3), ("inputdims", 1)])

    # `names` length must be 2
    y = x.nonzero(names=("a", "b"))
    assert 3 == y.size("a")
    assert y.shape == OrderedDict([("a", 3), ("b", 1)])

    # 2d tensor
    x = ntorch.tensor(
        [
            [0.6, 0.0, 0.0, 0.0],
            [0.0, 0.4, 0.0, 0.0],
            [0.0, 0.0, 1.2, 0.0],
            [2.0, 0.0, 0.0, -0.4],
        ],
        names=("alpha", "beta"),
    )
    y = x.nonzero()
    assert 5 == y.size("elements")
    assert x.shape == OrderedDict([("alpha", 4), ("beta", 4)])
    assert y.shape == OrderedDict([("elements", 5), ("inputdims", 2)])

    # `names` length must be 2
    y = x.nonzero(names=("a", "b"))
    assert 5 == y.size("a")
    assert y.shape == OrderedDict([("a", 5), ("b", 2)])


@pytest.mark.xfail
def test_nonzero_names():

    # `names` length must be 2
    x = ntorch.tensor([0, 1, 2, 0, 5], names=("dim",))
    y = x.nonzero(names=("a",))
    assert 2 == len(y.shape)

    # `names` length must be 2
    x = ntorch.tensor([0, 1, 2, 0, 5], names=("dim",))
    y = x.nonzero(names=("a", "b", "c"))
    assert 2 == len(y.shape)


# def test_log_softmax():
#     base = (
#         ntorch.tensor([0, 1, 2, 0, 5], names=("dim",))
#         .float()
#         .log_softmax("dim")
#     )
#     y = F.log_softmax(torch.tensor([0, 1, 2, 0, 5]).float(), dim=0)
#     expected = ntorch.tensor(y, names=("dim",))
#     assert_match(base, expected)


def test_indexing_basic():
    base = ntorch.randn(10, 2, 50, names=("alpha", "beta", "gamma"))

    base1 = base[{"alpha": 2}]
    assert base1.shape == OrderedDict([("beta", 2), ("gamma", 50)])

    base1 = base[{"beta": 0}]
    assert base1.shape == OrderedDict([("alpha", 10), ("gamma", 50)])

    base1 = base[{"alpha": slice(2, 5)}]
    assert base1.shape == OrderedDict(
        [("alpha", 3), ("beta", 2), ("gamma", 50)]
    )


def test_topk():
    k = 5
    base_torch = torch.rand([10, 10])
    dim_names = ("dim1", "dim2")
    base = ntorch.tensor(base_torch, names=dim_names).float().topk("dim2", k)
    topk, argtopk = base_torch.topk(k, dim=1)
    expected = (
        ntorch.tensor(topk, names=dim_names),
        ntorch.tensor(argtopk, names=dim_names),
    )
    assert_match(base[0], expected[0])
    assert_match(base[1], expected[1])


def test_chunk():
    base_torch = torch.rand([10, 10])
    dim_names = ("dim1", "dim2")
    named_chunks = ntorch.tensor(base_torch, names=dim_names).chunk(3, "dim1")
    torch_chunks = base_torch.chunk(3, 0)
    expected = tuple(named_chunks[0]._new(chunk) for chunk in torch_chunks)
    for i in range(len(torch_chunks)):
        assert named_chunks[i] == expected[i]


def test_index_set():
    base = ntorch.randn(10, 2, 50, names=("alpha", "beta", "gamma"))
    new = ntorch.randn(2, 50, names=("beta", "gamma"))
    base[{"alpha": 2}] = new
    new = ntorch.randn(3, 2, 50, names=("alpha", "beta", "gamma"))
    base[{"alpha": slice(0, 3)}] = new


def test_index_tensor():
    base = ntorch.zeros(10, 2, 50, names=("alpha", "beta", "gamma"))
    indices = ntorch.tensor([1, 2, 3, 4], names=("indices"))
    base1 = base[{"gamma": indices}]
    assert base1.shape == OrderedDict(
        [("alpha", 10), ("beta", 2), ("indices", 4)]
    )

    indices = ntorch.tensor([1, 2, 3, 4], names=("indices"))
    base1 = base[{"alpha": 1, "gamma": indices}]
    assert base1.shape == OrderedDict([("beta", 2), ("indices", 4)])

    indices = ntorch.tensor([[1, 2, 3], [1, 2, 3]], names=("d", "indices"))
    base1 = base[{"gamma": indices}]
    assert base1.shape == OrderedDict(
        [("alpha", 10), ("beta", 2), ("d", 2), ("indices", 3)]
    )


def test_setindex_tensor():
    base = ntorch.zeros(10, 2, 50, names=("alpha", "beta", "gamma")).float()
    indices = ntorch.tensor([1, 2, 3, 4], names=("indices")).long()
    vals = ntorch.tensor([52, 23.0, 42.9, 4.2], names=("indices")).float()
    b = base[{"alpha": 1, "beta": 1}]
    b[{"gamma": indices}] = vals
    assert base[{"alpha": 1, "beta": 1, "gamma": 1}].values == 52

    base[{"gamma": indices}] = 2


@pytest.mark.xfail
def test_unique_names():
    base = torch.zeros([10, 2])
    assert ntorch.tensor(base, ("alpha", "beta", "alpha"))


def test_names():
    base = torch.zeros([10, 2, 50])
    assert ntorch.tensor(base, ("alpha", "beta", "gamma"))


@pytest.mark.xfail
def test_bad_names():
    base = torch.zeros([10, 2])
    assert ntorch.tensor(base, ("elements_dim", "input_dims"))
