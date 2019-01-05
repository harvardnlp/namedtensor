import torch.nn.functional as F
from .core import NamedTensorCore, assert_match
import opt_einsum as oe

# Torch Ops
# Return a tensor of the same dimensions
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

_noshift_dim = {}

# Return a non-tensor info object
_info = {
    "dim",
    "is_contigious",
    "is_pinned",
    "size",
    "storage",
    "storage_offset",
    "storage_offset",
    "tolist",
    "stride",
    "all",
    "any",
}


# Takes a dim arg and reduces it.
_reduce = {
    "argmax",
    "argmin",
    "cumprod",
    "cumsum",
    "logsumexp",
    "mean",
    "median",
    "norm",
    "prod",
    "squeeze",
    "std",
    "sum",
}

_reduce_multi = {"min", "max", "unbind"}


# Broadcast and apply.
_binop = {
    "add",
    "masked_fill",
    "sub",
    "div",
    "mul",
    "eq",
    "ne",
    "lt",
    "gt",
    "le",
    "ge",
    "type_as",
}


def build(init, names, *args, **kwargs):
    tensor = init(tuple(names.values()), *args, **kwargs)
    names = tuple(names.keys())
    return NamedTensor(tensor, names)


def contract(names, *tensors):
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
    names = names.split()
    keep = [n for n in seen_names if n not in names]
    args.append([ids[n] for n in keep])
    return NamedTensor(oe.contract(*args, backend="torch"), keep)


class NamedTensor(NamedTensorCore):
    def index_select(self, name, index):
        new_names = []
        sizes = []
        for n in self._schema._names:
            if n == name:
                for n2 in index._schema._names:
                    new_names.append(n2)
                    sizes.append(index._size(n2))
            else:
                new_names.append(n)
                sizes.append(self._size(n))
        return NamedTensor(
            self._tensor.index_select(
                self._schema.get(name), index._tensor.view(-1)
            ).view(*sizes),
            new_names,
        )

    def narrow(self, change, start, end):
        return self._new(
            self._tensor.narrow(
                self._schema.get(change.split("-")[0].strip()), start, end
            ),
            updates=change,
        )

    def softmax(self, name):
        return self._new(F.softmax(self._tensor, dim=self._schema.get(name)))

    def logsoftmax(self, name):
        return self._new(F.logsoftmax(self.tensor, dim=self._schema.get(name)))

    def get(self, name, idx):
        results = self.access(name)[idx]
        return self._new(results, name)

    def sort(self, name):
        results = self._tensor.sort(self._schema.get(name))
        return tuple((self._new(r) for r in results))

    def unbind(self, name):
        results = self._tensor.unbind(self._schema.get(name))
        return tuple((self._new(r, name) for r in results))

    def max(self, name):
        results = self._tensor.max(self._schema.get(name))
        return tuple((self._new(r) for r in results))

    def min(self, name):
        results = self._tensor.max(self._schema.get(name))
        return tuple((self._new(r) for r in results))

    def renorm(self, p, name, maxnorm):
        results = self._tensor.renorm(p, self.get(name), maxnorm)
        return self._new(results)

    def access(self, dims):
        term = " ".join(
            dims.split() + [d for d in self._schema._names if d not in dims]
        )
        return self._rearrange(term)._tensor

    def op(self, axis_op, dim=None, shift=None):
        kwargs = {}
        if dim is not None:
            kwargs["dim"] = self._schema.get(dim)
        return self._new(axis_op(self._tensor, **kwargs), updates=shift)

    def __add__(self, b):
        return self.add(b)

    def __sub__(self, b):
        return self.sub(b)

    def __mul__(self, b):
        return self.mul(b)

    def __div__(self, b):
        return self.div(b)

    def __eq__(self, b):
        return self.eq(b)

    def __ne__(self, b):
        return self.ne(b)

    def __lt__(self, b):
        return self.lt(b)

    def __gt__(self, b):
        return self.gt(b)

    def __le__(self, b):
        return self.le(b)

    def __ge__(self, b):
        return self.ge(b)

    def __getattr__(self, methodname):
        if methodname in dir(self._tensor):
            method = getattr(self._tensor, methodname)
            if methodname in _noshift:
                # Call and wrap
                def call(*args, **kwargs):
                    return self._new(method(*args, **kwargs))

            elif methodname in _noshift_dim:

                def call(dim, *args, **kwargs):
                    return self._new(
                        method(self._schema.get(dim), *args, **kwargs)
                    )

            elif methodname in _info:
                # Call and return
                call = method

            elif methodname in _reduce:
                # Call, replace, and wrap
                def call(dim, *args, **kwargs):
                    cur = self
                    method = getattr(self._tensor, methodname)
                    for d in dim.split():
                        cur = cur._new(
                            method(cur._schema.get(d), *args, **kwargs), d
                        )
                        method = getattr(cur._tensor, methodname)
                    return cur

            elif methodname in _reduce_multi:

                def call(dim, *args, **kwargs):
                    method = getattr(self._tensor, methodname)
                    results = method(self._schema.get(dim), *args, **kwargs)
                    return tuple((self._new(r, dim) for r in results))

            elif methodname in _binop:

                def call(other, *args):
                    if isinstance(other, NamedTensor):
                        b = other
                        order = self._broadcast_order(b)
                        a1 = self._force_order(order)
                        b1 = b._force_order(order)
                        method = getattr(a1._tensor, methodname)
                        assert_match(a1, b1)
                        return a1._new(method(b1._tensor, *args))
                    else:
                        method = getattr(self._tensor, methodname)
                        return self._new(method(other, *args))

            else:
                assert False, "Method not implemented"
            return call
        assert False, "Method does not exist"

    def contract(self, names, *others):
        "Contract dimension `names` with each of the other tensors"
        return contract(names, *((self,) + others))
