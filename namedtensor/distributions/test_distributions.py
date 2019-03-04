from .. import ntorch
from . import NamedDistribution, ndistributions
import torch
from collections import OrderedDict
import torch.distributions as ds


def test_shift():
    dist = ds.Normal(torch.randn([10, 10]), torch.randn([10, 10]))

    q = NamedDistribution(dist, ("batch1", "batch2"), ())

    q.sample((10,), ("sample1",))


def test_build():
    a = ntorch.randn(10, 20, names=("batch1", "batch2"))
    dist = ndistributions.Normal(a, a)
    s = dist.sample((30,), ("sample1",))
    assert s.shape == OrderedDict(
        [("sample1", 30), ("batch1", 10), ("batch2", 20)]
    )

    s = dist.sample()
    assert s.shape == OrderedDict([("batch1", 10), ("batch2", 20)])

    s = dist.sample((30, 40), names=("sample1", "sample2"))
    assert s.shape == OrderedDict(
        [("sample1", 30), ("sample2", 40), ("batch1", 10), ("batch2", 20)]
    )

    assert dist.batch_shape == OrderedDict([("batch1", 10), ("batch2", 20)])

    out = dist.log_prob(
        ntorch.randn(
            10, 25, 1, 1, names=("sample1", "sample2", "batch1", "batch2")
        )
    )
    assert out.shape == OrderedDict(
        [("sample1", 10), ("sample2", 25), ("batch1", 10), ("batch2", 20)]
    )

    out = dist.entropy()
    print(out.values.size())
    assert out.shape == OrderedDict([("batch1", 10), ("batch2", 20)])


def test_multi():
    mean = ntorch.randn(10, 20, 30, names=("batch1", "batch2", "m"))
    sigma = ntorch.ones(10, 20, 30, 30, names=("batch1", "batch2", "v1", "v2"))
    sigma.values[:, :] = torch.eye(30)
    dist = ndistributions.MultivariateNormal(
        mean, sigma, event_dims=("m",), scale_dims=("v1", "v2")
    )

    assert dist.batch_shape == OrderedDict([("batch1", 10), ("batch2", 20)])
    print(dist.event_shape)
    assert dist.event_shape == OrderedDict([("m", 30)])


def test_cat():
    logits = ntorch.randn(10, 20, 30, names=("batch1", "batch2", "logits"))
    dist = ndistributions.Categorical(logits=logits, logit_dim="logits")
    assert dist.batch_shape == OrderedDict([("batch1", 10), ("batch2", 20)])
    assert dist.event_shape == OrderedDict([])

    s = dist.sample((30, 40), names=("sample1", "sample2"))
    assert s.shape == OrderedDict(
        [("sample1", 30), ("sample2", 40), ("batch1", 10), ("batch2", 20)]
    )
