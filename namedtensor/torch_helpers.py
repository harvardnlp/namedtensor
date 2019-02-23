import torch.nn.functional as F
import torch
from .core import NamedTensorBase, assert_match
from .utils import make_tuple


class NamedTensor(NamedTensorBase):
    def __getitem__(self, index):
        if isinstance(index, dict):
            cur = self
            for k, v in index.items():
                if isinstance(v, slice):
                    cur = cur.narrow(k, v.start, v.stop - v.start)
                elif isinstance(v, NamedTensor):
                    cur = cur.index_select(k, v)
                else:
                    cur = cur.get(k, v)
            return cur
        elif isinstance(index, NamedTensor):
            if (
                index.type() == "torch.ByteTensor"
                or index.type() == "torch.cuda.ByteTensor"
            ):
                return self.masked_select(index)
            raise RuntimeError("Masked namedtensor must be byte tensor.")
        else:
            raise RuntimeError("Index must be dict or namedtensor.")

    def __setitem__(self, index, val):

        if isinstance(val, NamedTensor):
            copy = True
        else:
            copy = False

        if isinstance(index, dict):
            cur = self
            for k, v in index.items():
                if isinstance(v, slice):
                    cur = cur.narrow(k, v.start, v.stop - v.start)
                elif isinstance(v, NamedTensor):
                    assert len(index) == 1
                    if copy:
                        cur.index_copy_(k, v, val)
                    else:
                        cur.index_fill_(k, v, val)
                    return self
                else:
                    cur = cur.get(k, v)
            if copy:
                cur.copy_(val)
            else:
                cur.fill_(val)
        elif isinstance(index, NamedTensor):
            if (
                index.type() == "torch.ByteTensor"
                or index.type() == "torch.cuda.ByteTensor"
            ):
                if copy:
                    return self.masked_scatter_(index, val)
                else:
                    return self.masked_fill_(index, val)
            raise RuntimeError("Masked namedtensor must be byte tensor.")
        else:
            raise RuntimeError("Index must be dict or namedtensor.")
        return self

    def copy_(self, other):
        return self._setter(other, "copy_")

    def _setter(self, other, method, vals=[]):
        order = other._mask_broadcast_order(self)
        other = other._force_order(order)

        args = [other.values] + vals
        getattr(self.values, method)(*args)
        return self

    def get(self, name, idx):
        "Returns a namedtensor by indexing into dim name"
        dim = self._schema.get(name)
        return self._new(
            self.values.narrow(dim, torch.tensor(idx), 1).squeeze(dim), name
        )

    def renorm(self, p, name, maxnorm):
        "Apply :py:meth:`torch.Tensor.renorm` over `name`"
        results = self._tensor.renorm(p, self.get(name), maxnorm)
        return self._new(results)

    def dot(self, names, *others):
        "Contract dimension `names` with each of the other tensors"
        from .torch_base import ntorch

        return ntorch.dot(names, *((self,) + others))

    # def access(self, dims):
    #     term = dims.split() + [d for d in self._schema._names if d not in dims]
    #     return self.transpose(*term)._tensor

    # def debug(self):
    #     print(self.shape)
    #     return self

    def augment(self, axis_op, add, dim=None, **kwargs):
        return self.op(axis_op, dim=dim, _add=add, **kwargs)

    def reduce(self, axis_op, reduced, dim=None, **kwargs):
        return self.op(axis_op, dim=dim, _drop=reduced, **kwargs)

    def reduce2(self, other, axis_op, reduced, dim=None, **kwargs):
        return self.op2(other, axis_op, dim=dim, _drop=reduced, **kwargs)

    def op(self, *axis_ops, dim=None, _drop=None, _add=None, **kwargs):
        "Apply ops that may change dimensions sizes "
        func_args = {}
        if dim is not None:
            func_args["dim"] = self._schema.get(dim)
        _drop = make_tuple(_drop)
        for v in _drop:
            self._schema.get(v)

        cur = self._tensor
        for axis_op in axis_ops:
            cur = axis_op(cur, **func_args)

        for k, vs in kwargs.items():
            for v in make_tuple(vs):
                self._schema.get(v)

        if _add is None and _drop is None:
            assert len(cur.shape) == len(
                self._tensor.shape
            ), "In shape %s, Out shape %s" % (cur.shape, self._tensor.shape)

        out = self._new(
            cur,
            drop=_drop,
            add=make_tuple(_add),
            updates={
                (v[0] if isinstance(v, tuple) else v): k
                for k, v in kwargs.items()
            },
        )

        # for k, v in self.shape.items():
        #     assert k not in out.shape or v == out.shape[k], (
        #         "name needs to change for updated dimensions"
        #         + str(axis_ops)
        #         + str(k)
        #     )
        return out

    def op2(self, y, axis_op, dim=None, _drop=None, **kwargs):
        return self.op(lambda x: axis_op(x, y.values), _drop=_drop, **kwargs)

    def __neg__(self):
        return self.neg()

    def __add__(self, b):
        return self.add(b)

    def __radd__(self, b):
        return self.add(b)

    def __sub__(self, b):
        return self.sub(b)

    def __rsub__(self, b):
        return -self.sub(b)

    def __mul__(self, b):
        return self.mul(b)

    def __rmul__(self, b):
        return self.mul(b)

    def __div__(self, b):
        return self.div(b)

    def __truediv__(self, b):
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
            if methodname in self._noshift | self._noshift_args:

                def call(*args, **kwargs):
                    return self._new(method(*args, **kwargs))

                call.__doc__ = method.__doc__
            elif methodname in self._noshift_nn:
                method = getattr(F, methodname)

                def call(*args, **kwargs):
                    return self._new(method(self.values, *args, **kwargs))

                call.__doc__ = method.__doc__

            elif methodname in self._noshift_dim:

                def call(dim, *args, **kwargs):
                    return self._new(
                        method(self._schema.get(dim), *args, **kwargs)
                    )

                call.__doc__ = method.__doc__
            elif methodname in self._noshift_nn_dim:
                method = getattr(F, methodname)

                def call(dim, *args, **kwargs):
                    return self._new(
                        method(
                            self.values,
                            dim=self._schema.get(dim),
                            *args,
                            **kwargs
                        )
                    )

                call.__doc__ = method.__doc__

            elif methodname in self._inline:

                def call(*args, **kwargs):
                    method(*args, **kwargs)
                    return self

                call.__doc__ = method.__doc__

            elif methodname in self._info:
                call = method
            elif methodname in self._reduce:

                def call(dim=None, *args, **kwargs):
                    cur = self
                    method = getattr(cur._tensor, methodname)
                    if dim is None:
                        return NamedTensor(method(*args, **kwargs), ())
                    dim = make_tuple(dim)
                    method = getattr(self._tensor, methodname)
                    for d in dim:
                        cur = cur._new(
                            method(cur._schema.get(d), *args, **kwargs), d
                        )
                        method = getattr(cur._tensor, methodname)
                    return cur

                call.__doc__ = self._reduce_doc + method.__doc__
            elif methodname in self._reduce_multi:

                def call(dim, *args, **kwargs):
                    method = getattr(self._tensor, methodname)
                    results = method(self._schema.get(dim), *args, **kwargs)
                    return tuple((self._new(r, dim) for r in results))

                call.__doc__ = self._reduce_doc + method.__doc__

            elif methodname in self._core:
                from .torch_base import ntorch

                method = getattr(ntorch, methodname)

                def call(*args, **kwargs):
                    return method(self, *args, **kwargs)

                call.__doc__ = method.__doc__

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

                call.__doc__ = method.__doc__
            else:
                raise NotImplementedError(methodname)

            return call
        raise NotImplementedError(methodname)

    def __dir__(self):
        return (
            set(self.__class__.__dict__.keys())
            | self._noshift
            | self._noshift_args
            | self._noshift_nn
            | self._info
            | self._reduce
            | self._reduce_multi
            | self._binop
            | self._inline
            | self._core
        )

    # Torch Ops
    # Return a tensor of the same dimensions
    _noshift = {
        "abs",
        "acos",
        "asin",
        "atan",
        "byte",
        "ceil",
        "clone",
        "contiguous",
        "cos",
        "cosh",
        "cpu",
        "cuda",
        "detach",
        "double",
        "exp",
        "expm1",
        "float",
        "floor",
        "frac",
        "half",
        "int",
        "long",
        "log",
        "relu",
        "round",
        "rsqrt",
        "short",
        "sigmoid",
        "sign",
        "sin",
        "sinh",
        "sqrt",
        "neg",
        "to",
        "tan",
        "tanh",
        "trunc"
    }

    _noshift_args = {"tril", "triu", "pow", "fmod", "clamp", "reciprical"}

    _noshift_nn = {"relu"}

    _noshift_nn_dim = {"softmax", "log_softmax"}

    _noshift_dim = {"cumprod", "cumsum"}

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
        "backward",
        "numpy",
        "item",
        "type",
    }

    _reduce_doc = """
NamedTensor modifies this method to take a named `dim` as
the argument instead of a dimension index. Otherwise
doc is the same as below.

====================
    """

    # Takes a dim arg and reduces it.
    _reduce = {
        "argmax",
        "argmin",
        "logsumexp",
        "mean",
        "prod",
        "std",
        "sum",
        "squeeze",
    }

    _reduce_multi = {"min", "max", "sort", "unbind", "median"}

    _extra = {"masked_fill", "type_as"}

    # Broadcast and apply.
    _binop = {"add", "sub", "div", "mul", "eq", "ne", "lt", "gt", "le", "ge"}

    # Inline.
    _inline = {
        "fill_",
        "random_",
        "abs_",
        "acos_",
        "asin_",
        "atan_",
        "ceil_",
        "clamp_",
        "cos_",
        "cosh_",
        "exp_",
        "floor_",
        "fmod_",
        "log_",
        "pow_",
        "round_",
        "rsqrt_",
        "sigmoid_",
        "sign_",
        "sin_",
        "sinh_",
        "sqrt_",
        "sub_",
        "tan_",
        "tanh_",
    }

    _core = {
        "gather",
        "nonzero",
        "scatter_",
        "narrow",
        "masked_select",
        "masked_scatter",
        "masked_fill_",
        "index_select",
        "index_copy_",
        "index_fill_",
        "topk",
    }

    # def gather(self, dim, index, index_dim):
    #     """
    #     Apply gather where `self_dim` is reduced out
    #     based on `index` from `index_dim`.
    #     """
    #     from .torch_base import ntorch

    #     return ntorch.gather(self, dim, index, index_dim)

    # def scatter_(self, dim, index, src, index_dim):
    #     """
    #     Apply scatter where `dim` gets the
    #     scattered values of `src` based in `index` along `index_dim`.
    #     """

    #     from .torch_base import ntorch

    #     ntorch.scatter_(self, dim, index, src, index_dim)

    # def narrow(self, name, start, end):
    #     "Narrow into the `kwargs` dimension and rename it"
    #     from .torch_base import ntorch

    #     return ntorch.narrow(self, name, start, end)

    # def masked_select(self, mask, name="on"):
    #     "Applies `mask` and returns a 1D tensor with name `name`"
    #     from .torch_base import ntorch

    #     return ntorch.masked_select(self, mask, name)

    # def masked_fill_(self, mask, val):
    #     from .torch_base import ntorch

    #     return ntorch.masked_fill_(self, mask, val)

    # def masked_scatter_(self, mask, source):
    #     from .torch_base import ntorch

    #     return ntorch.masked_scatter_(self, mask, source)
    # def relu(self):
    #     "Apply relu"
    #     return self._new(F.relu(self._tensor))

    # def softmax(self, name):
    #     "Apply softmax over dim `name`"
    #     return self._new(F.softmax(self._tensor, dim=self._schema.get(name)))

    # def log_softmax(self, name):
    #     "Apply log softmax over dim `name`"
    #     return self._new(
    #         F.log_softmax(self._tensor, dim=self._schema.get(name))
    #     )
