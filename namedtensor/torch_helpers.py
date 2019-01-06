import torch.nn.functional as F
from .core import NamedTensorBase, assert_match


class NamedTensor(NamedTensorBase):
    def index_select(self, name, index):
        "Index into dimension names with the `index` named tensors."
        new_names = []
        sizes = []
        for n in self._schema._names:
            if n == name:
                for n2 in index._schema._names:
                    new_names.append(n2)
                    sizes.append(index.size(n2))
            else:
                new_names.append(n)
                sizes.append(self.size(n))
        return NamedTensor(
            self._tensor.index_select(
                self._schema.get(name), index._tensor.view(-1)
            ).view(*sizes),
            new_names,
        )

    def dot(self, names, *others):
        "Contract dimension `names` with each of the other tensors"
        from .torch_base import ntorch

        return ntorch.dot(names, *((self,) + others))

    def narrow(self, start, end, **kwargs):
        "Narrow into the `kwargs` dimension and rename it"
        from .torch_base import ntorch

        return ntorch.narrow(self, start, end, **kwargs)

    def softmax(self, name):
        "Apply softmax over dim `name`"
        return self._new(F.softmax(self._tensor, dim=self._schema.get(name)))

    def logsoftmax(self, name):
        "Apply log softmax over dim `name`"
        return self._new(F.logsoftmax(self.tensor, dim=self._schema.get(name)))

    def get(self, name, idx):
        results = self.access(name)[idx]
        return self._new(results, name)

    def renorm(self, p, name, maxnorm):
        "Apply :py:meth:`torch.Tensor.renorm` over `name`"
        results = self._tensor.renorm(p, self.get(name), maxnorm)
        return self._new(results)

    def access(self, dims):
        term = dims.split() + [d for d in self._schema._names if d not in dims]
        return self.transpose(*term)._tensor

    def op(self, axis_op, dim=None, **kwargs):
        kwargs = {}
        if dim is not None:
            kwargs["dim"] = self._schema.get(dim)
        return self._new(axis_op(self._tensor, **kwargs), updates=kwargs)

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
            if methodname in self._noshift:
                # Call and wrap
                def call(*args, **kwargs):
                    return self._new(method(*args, **kwargs))

            elif methodname in self._noshift_dim:

                def call(dim, *args, **kwargs):
                    return self._new(
                        method(self._schema.get(dim), *args, **kwargs)
                    )

            elif methodname in self._info:
                # Call and return
                call = method

            elif methodname in self._reduce:
                # Call, replace, and wrap
                def call(dim, *args, **kwargs):
                    cur = self
                    if not isinstance(dim, tuple):
                        dim = (dim,)
                    method = getattr(self._tensor, methodname)
                    for d in dim:
                        cur = cur._new(
                            method(cur._schema.get(d), *args, **kwargs), d
                        )
                        method = getattr(cur._tensor, methodname)
                    return cur

            elif methodname in self._reduce_multi:

                def call(dim, *args, **kwargs):
                    method = getattr(self._tensor, methodname)
                    results = method(self._schema.get(dim), *args, **kwargs)
                    return tuple((self._new(r, dim) for r in results))

            elif methodname in self._binop:

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

    _reduce_multi = {"min", "max", "sort", "unbind"}

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
