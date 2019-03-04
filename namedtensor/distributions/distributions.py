from ..schema import _Schema
from ..torch_helpers import NamedTensor
import torch
import torch.distributions


class NamedDistribution:
    def __init__(self, dist, batch_names, event_names):
        self._dist = dist
        self._batch_schema = _Schema.build(batch_names, 0)
        self._event_schema = _Schema.build(event_names, 0)

    @staticmethod
    def build(init, *args, **kwargs):
        collect = []

        def fix(v):
            if isinstance(v, NamedTensor):
                collect.append(v)
                return v.values
            else:
                return v

        drop = {"dims_event", "dims_scale", "dim_logit"}
        event_dims = kwargs.get("dims_event", [])
        scale_dims = kwargs.get("dims_scale", [])
        logit_dim = kwargs.get("dim_logit", "")
        args = list(args)
        if "dims_scale" in kwargs:
            args[1] = args[1].transpose(*kwargs["dims_scale"])
        if "dim_logit" in kwargs:
            if "logits" in kwargs:
                kwargs["logits"] = kwargs["logits"].transpose(logit_dim)
            else:
                args[0] = args[0].transpose(logit_dim)

        new_args = [fix(v) for v in args]
        new_kwargs = {k: fix(v) for k, v in kwargs.items() if k not in drop}
        dist = init(*new_args, **new_kwargs)

        c = collect[0]
        return NamedDistribution(
            dist,
            [
                n
                for n in c._schema._names
                if n not in event_dims
                and n not in scale_dims
                and n != logit_dim
            ],
            event_dims,
        )

    @property
    def batch_shape(self):
        "Named batch shape as an ordered dict"
        return self._batch_schema.ordered_dict(self._dist.batch_shape)

    @property
    def event_shape(self):
        "Named event shape as an ordered dict"
        return self._event_schema.ordered_dict(self._dist.event_shape)

    def _sample(self, fn, sizes, names):
        tensor = fn(torch.Size(sizes))
        return NamedTensor(
            tensor,
            names + self._batch_schema._names + self._event_schema._names,
        )

    def sample(self, sizes=(), names=()):
        return self._sample(self._dist.sample, sizes, names)

    def rsample(self, sizes=(), names=()):
        return self._sample(self._dist.rsample, sizes, names)

    def __getattr__(self, name):
        if name in self._batch_methods:

            def call():
                method = getattr(self._dist, name)
                return NamedTensor(method(), self._batch_schema)

            return call
        elif name in self._batch:
            method = getattr(self._dist, name)
            return NamedTensor(method, self._batch_schema)
        elif name in self._properties:
            return getattr(self._dist, name)
        elif name in self._bin:

            def call(values):
                method = getattr(self._dist, name)
                print(values.values.size())
                return NamedTensor(
                    method(values.values),
                    values._schema._names[-len(self._event_schema._names) :],
                )

            return call
        assert False, "No attr"

    def __repr__(self):
        return repr(self._dist)

    # batch shape methods
    _batch_methods = {"entropy", "perplexity"}

    # batch shape properties
    _batch = {"mean", "stddev", "variance"}

    # properties
    _properties = {"arg_constraints", "support"}

    # batch shape methods
    _bin = {"log_prob", "icdf", "cdf"}


class NDistributions(type):
    def __getattr__(cls, name):
        if name in cls._build:

            def call(*args, **kwargs):
                return NamedDistribution.build(
                    getattr(torch.distributions, name), *args, **kwargs
                )

            return call
        elif name in cls._other:

            def call(*args, **kwargs):
                new_args = [arg._dist for arg in args]
                return getattr(torch.distributions, name)(*new_args, **kwargs)

            return call

        assert False, "Function does not exist"

    _build = {
        "Normal",
        "Multinomial",
        "Bernoulli",
        "Beta",
        "Binomial",
        "Categorical",
        "Cauchy",
        "Chi2",
        "Dirichlet",
        "Exponential",
        "FisherSnedecor",
        "Gamma",
        "Geometric",
        "Gumbel",
        "HalfCauchy",
        "HalfNormal",
        "Independent",
        "Laplace",
        "LogNormal",
        "LowRankMultivariateNormal",
        "Multinomial",
        "MultivariateNormal",
        "NegativeBinomial",
        "Normal",
        "OneHotCategorical",
        "Pareto",
        "Poisson",
        "RelaxedBernoulli",
        "RelaxedOneHotCategorical",
        "StudentT",
        "TransformedDistribution",
        "Uniform",
        "Weibull",
    }

    _other = {"kl_divergence"}


class ndistributions(metaclass=NDistributions):
    pass
