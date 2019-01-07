import torch
from .torch_helpers import NamedTensor
import opt_einsum as oe


def make_tuple(names):
    if isinstance(names, tuple):
        return names
    else:
        return (names,)


class NTorch(type):
    def __getattr__(cls, name):
        if name in cls._build:

            def call(names, *args, **kwargs):
                return cls.build(getattr(torch, name), names, *args, **kwargs)

            return call
        elif name in cls._noshift:

            def call(ntensor, *args, **kwargs):
                return getattr(ntensor, name)(*args, **kwargs)

            return call
        raise NotImplementedError(name)

    @classmethod
    def dot(cls, names, *tensors):
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
        args.append([ids[n] for n in keep])
        return cls.tensor(oe.contract(*args, backend="torch"), keep)

    @staticmethod
    def narrow(tensor1, start, end, **kwargs):
        value, key = next(iter(kwargs.items()))
        return tensor1._new(
            tensor1._tensor.narrow(tensor1._schema.get(key), start, end),
            updates={v:k for k,v in kwargs.items()}
        )


    @staticmethod
    def cat(tensors, dim):
        dim = tensors[0]._schema.get(dim)
        for t in tensors[1:]:
            assert t._schema._names == tensors[0]._schema._names
        return tensors[0]._new(torch.cat([t.values for t in tensors],
                                         dim=dim))

    @staticmethod
    def build(init, names, *args, **kwargs):
        tensor = init(*tuple(names.values()), *args, **kwargs)
        names = tuple(names.keys())
        return NamedTensor(tensor, names)

    @staticmethod
    def tensor(*args, **kwargs):
        return NamedTensor(*args, **kwargs)

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

    pass
