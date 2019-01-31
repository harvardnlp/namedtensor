import torch
from .torch_helpers import NamedTensor
from . import torch_nn
import opt_einsum as oe


def make_tuple(names):
    if names is None:
        return ()

    if isinstance(names, tuple):
        return names
    else:
        return (names,)


class NTorch(type):
    def __getattr__(cls, name):
        if name in cls._build:

            def call(*args, **kwargs):
                assert "names" in kwargs, "Need `names` kwarg."
                names = kwargs["names"]
                del kwargs["names"]
                return cls.build(getattr(torch, name), names, *args, **kwargs)

            call.__doc__ = getattr(torch, name).__doc__
            return call
        elif name in cls._noshift:

            def call(ntensor, *args, **kwargs):
                return getattr(ntensor, name)(*args, **kwargs)

            call.__doc__ = getattr(torch, name).__doc__
            return call
        raise NotImplementedError(name)

    @classmethod
    def dot(cls, dims, *tensors):
        names = dims
        args = []
        ids = {}
        seen_names = []
        for t in tensors:
            group = []
            for name in t._schema._names:
                if name not in ids:
                    ids[name] = len(ids)
                    seen_names.append(name)
                group.append(ids[name])
            args.append(t._tensor)
            args.append(group)
        names = make_tuple(names)
        keep = [n for n in seen_names if n not in names]
        for n in names:
            if n not in seen_names:
                raise RuntimeError("No dimension %s to contract along" % n)
        args.append([ids[n] for n in keep])
        return cls.tensor(oe.contract(*args, backend="torch"), keep)

    @staticmethod
    def narrow(tensor1, dim, start, end):
        name = dim
        return tensor1._new(
            tensor1._tensor.narrow(tensor1._schema.get(name), start, end)
        )

    @staticmethod
    def cat(tensors, dim):
        "Concate a list of named tensors along dim."
        dim = tensors[0]._schema.get(dim)
        for t in tensors[1:]:
            assert t._schema._names == tensors[0]._schema._names
        return tensors[0]._new(torch.cat([t.values for t in tensors], dim=dim))

    @staticmethod
    def gather(input, index, **kwargs):
        outdim = tuple(kwargs.keys())[0]
        indim = kwargs[outdim]
        index_order = [
            (n if n != indim else outdim) for n in input._schema._names
        ]
        b1 = index._force_order(index_order)
        dim = input._schema.get(indim)
        return input._new(input.values.gather(dim, b1.values), updates=kwargs)

    @staticmethod
    def masked_select(input, mask, dim):
        order = input._broadcast_order(mask)
        a1 = input._force_order(order)
        b1 = mask._force_order(order)
        return NamedTensor(a1.values.masked_select(b1.values), dim)

    @staticmethod
    def nonzero(tensor, names=("elements_dim", "input_dims")):
        """
        Returns a tensor containing the indices of all non-zero elements.

        Parameters
        ----------
        names : tuple, optional
            Names for the output dimensions
            default value: ("elements_dim", "input_dims")
            default output shape: OrderedDict([("elements_dim", number of non-zero elements),
                                                 ("input_dims", input tensor's number of dimensions)])
        """

        indices = torch.nonzero(tensor.values)
        return NamedTensor(tensor=indices, names=names)

    @staticmethod
    def scatter_(input, index, src, **kwargs):
        indim = tuple(kwargs.keys())[0]
        outdim = kwargs[indim]
        index_order = [
            (n if n != indim else outdim) for n in input._schema._names
        ]

        index_force = index._force_order(index_order)
        src_force = src._force_order(index_order)
        dim = input._schema.get(indim)
        input.values.scatter_(dim, index_force.values, src_force.values)

    @staticmethod
    def build(init, names, *args, **kwargs):
        tensor = init(*args, **kwargs)
        return NamedTensor(tensor, names)

    @staticmethod
    def tensor(*args, **kwargs):
        if isinstance(args[0], torch.Tensor):
            return NamedTensor(*args, **kwargs)
        else:
            return NamedTensor(
                *((torch.tensor(args[0]),) + args[1:]), **kwargs
            )

    @classmethod
    def __dir__(cls):
        return set(cls.__dict__.keys()) | cls._build | cls._noshift

    _build = {"ones", "zeros", "randn", "empty", "rand"}

    _noshift = {
        "abs",
        "acos",
        "asin",
        "atan",
        "byte",
        "ceil",
        "clamp",
        "clone",
        "contiguous",
        "cos",
        "cosh",
        "cpu",
        "cuda",
        "double",
        "exp",
        "expm1",
        "float",
        "floor",
        "fmod",
        "frac",
        "half",
        "int",
        "long",
        "log",
        "pow",
        "reciprical",
        "round",
        "rsqrt",
        "short",
        "sigmoid",
        "sign",
        "sin",
        "sinh",
        "sqrt",
        "sub",
        "to",
        "tan",
        "tanh",
        "tril",
        "triu",
        "trunc",
    }


class ntorch(metaclass=NTorch):
    nn = torch_nn
    pass
