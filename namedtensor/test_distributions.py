from . import assert_match, ntorch, NamedTensor, NamedDistribution, ndistributions
import torch
from collections import OrderedDict
import torch.distributions as ds

def test_shift():
    dist = ds.Normal(torch.randn([10, 10]), torch.randn([10, 10]))

    q = NamedDistribution(dist, ('batch1', 'batch2'), ())

    q.sample(sample1=10)



def test_build():
    a = ntorch.randn(dict(batch1=10, batch2=20))
    dist = ndistributions.Normal(a, a)
    s = dist.sample(sample1=30)
    assert s.shape == OrderedDict([("sample1", 30), ("batch1", 10), ("batch2", 20)])

    s = dist.sample()
    assert s.shape == OrderedDict([("batch1", 10), ("batch2", 20)])


    s = dist.sample(sample1=30, sample2=40)
    assert s.shape == OrderedDict([("sample1", 30), ("sample2", 40),
                                   ("batch1", 10), ("batch2", 20)])

    assert dist.batch_shape == OrderedDict([("batch1", 10), ("batch2", 20)])


    out = dist.log_prob(ntorch.randn(dict(sample1=10, sample2=25)))
    assert out.shape == OrderedDict([("batch1", 10), ("batch2", 20),
                                     ("sample1", 10), ("sample2", 25)])


    out = dist.entropy()
    assert out.shape == OrderedDict([("batch1", 10), ("batch2", 20)])


def test_build():
    mean = ntorch.randn(dict(batch1=10, batch2=20, m=30))
    sigma = ntorch.ones(dict(batch1=10, batch2=20, v1=30, v2=30))
    sigma.values[:, :] = torch.eye(30)
    dist = ndistributions.MultivariateNormal(mean, sigma)

    assert dist.batch_shape == OrderedDict([("batch1", 10), ("batch2", 20)])
    assert dist.event_shape == OrderedDict([("m", 30)])
