from einops import reduce, rearrange
from collections import OrderedDict
import re
import numpy as np

import torch
import torch.nn.functional as F
import opt_einsum as oe

# Torch Ops

# Return a tensor of the same dimensions
_noshift = {"abs", "acos",  "asin", "atan", "byte",
           "ceil", "clamp", "clone", "contiguous",
           "cos", "cosh", "cpu", "cuda", "double",
           "exp", "expm1", "float", "floor", "fmod",
           "frac", "half", "int", "long", "log", "pow",
           "reciprical", "round", "rsqrt", "short",
            "sigmoid", "sign", "sin", "sinh", "sqrt",
            "sub", "to", "tan", "tanh", "tril", "triu",
            "trunc"}

_noshift_dim = {}

# Return a non-tensor info object
_info = {"dim", "is_contigious", "is_pinned", "size",
        "storage", "storage_offset", "storage_offset",
         "tolist", "stride"}


# Takes a dim arg and reduces it.
_reduce = {"argmax", "argmin", "cumprod",
           "cumsum", "logsumexp", "max", "mean", "median",
           "min", "norm", "prod", "squeeze", "std",
           "sum"}

# Broadcast and apply.
_binop = {"add", "masked_fill", "sub", "div", "mul", "eq", "ne", "lt", "gt", "le", "ge", "type_as"}

#
#unknown = {"diag", "dist", "gather", "index_select", "scatter", "select", trace,  }


def contract(names, *tensors):
    args = []
    ids = {}
    seen_names = []
    for t in tensors:
        group = []
        for name in t._names:
            if name not in ids:
                ids[name] = len(ids)
                seen_names.append(name)
            group.append(ids[name])
        args.append(t.tensor)
        args.append(group)
    names = names.split()
    keep = [n for n in seen_names if n not in names]
    args.append([ids[n] for n in keep])
    return NamedTensor(oe.contract(*args, backend="torch"), keep)

def lift(fn, in_specs, out_spec):
    in_specs = [s.split() for s in in_specs]
    out_spec = out_spec.split()
    def lifted(*inputs):
        assert_match(inputs)
        assert len(inputs) == len(in_specs)
        lifted_inputs = []
        batch_dims = []
        for inp, spec in zip(inputs, in_specs):
            lifted_inputs.append(inp._promote(spec).tensor
                                 if spec is not None else inp)
            if spec is not None:
                batch_dims = [d for d in inp._names not in spec]
        out = fn(*lifted_inputs)
        return NamedTensor(out, batch_dims + out_spec)
    return lifted


def assert_match(*tensors):
    sizes = {}
    failure = False
    for t in tensors:
        for k, v in t._sizes.items():
            if v == 1: continue
            if k in sizes:
                failure = (failure or sizes[k] != v)
            else:
                sizes[k] = v
    assert not failure, " ".join([str(t._sizes) for t in tensors])


def build(init, names, **kwargs):
    tensor = init(tuple(names.values()), **kwargs)
    names = tuple(names.keys())
    return NamedTensor(tensor, names)


def randn(names, **kwargs):
    return build(torch.randn, names, **kwargs)
def ones(names, **kwargs):
    return build(torch.ones, names, **kwargs)
def zeros(names, **kwargs):
    return build(torch.zeros, names, **kwargs)


class NamedTensor:
    def __init__(self, tensor, names):
        if isinstance(names, str):
            names = names.split()
        self.tensor = tensor
        self._names = names
        shape = self.tensor.shape
        self._sizes = OrderedDict(((d, shape[i]) for i, d in enumerate(self._names)))
        self._axes = OrderedDict(((d, i) for i, d in enumerate(self._names)))

    def _new(self, tensor, drop=None, updates=None):
        update_dict = {}
        if updates is not None:
            for u in updates:
                group = re.match(r"(\w+) -> (\w+)", updates)
                start, end = group.groups()
                update_dict[start] = end

        return NamedTensor(tensor,
                           [update_dict.get(n, n) for n in self._names if n != drop])

    def _to_einops(self):
        return " ".join(self._names)

    @property
    def named_shape(self):
        return self._sizes

    def contract(self, names, *others):
        return contract(names, *((self,) + others))


    def unbind(self, name):
        results = self.tensor.unbind(self._axes[name])
        return tuple((self._new(r, name) for r in results))

    def get(self, name, idx):
        results = self.access(name)[idx]
        return self._new(results, name)



    def sort(self, name):
        results = self.tensor.sort(self._axes[name])
        return tuple((self._new(r) for r in results))

    def renorm(self, p, name, maxnorm):
        results = self.tensor.renorm(p, self._axes[name], maxnorm)
        return self._new(results)



    def shift(self, *ops, **kwargs):
        tensor = self
        for op in ops:
            if op.strip().startswith("("):
                tensor = tensor._merge(op)
            elif op.strip().endswith(")"):
                tensor = tensor._split(op, **kwargs)
            elif op.strip().startswith("..."):
                tensor = tensor._promote(op)
            else:
                tensor = tensor._rearrange(op)
        return tensor

    def _merge(self, mergestr):
        group = re.match(r"\(([\w+ ?]+)\) -> (\w+)", mergestr)
        shape = self.tensor.shape
        strnames, dim = group.groups()
        names = strnames.split()
        s = ""
        ex = ""
        first = True
        for d in self._names:
            if d not in names:
                s += " " + d
                ex += " " + d
            elif first:
                s += " (" + strnames + ")"
                ex += " " + dim
                first = False

        tensor = rearrange(self.tensor, "%s -> %s"%(self._to_einops(), s))
        return NamedTensor(tensor, ex)

    def _split(self, splitstr, **kwargs):
        group = re.match(r"(\w+) -> \(([\w+ ?]+)\)", splitstr)
        dim, strnames = group.groups()
        names = strnames.split()
        query = ""
        ex = ""
        for i, d in enumerate(self._names):
            if d != dim:
                query += " " + d
                ex += " " + d
            else:
                query += " (" + strnames + ")"
                ex += " " + strnames

        tensor = rearrange(self.tensor, "%s -> %s"%(query, ex),
                           **{d:kwargs[d] for d in names
                              if d in kwargs})
        return NamedTensor(tensor, ex)

    def _rearrange(self, term):
        assert ")" not in term
        recipe = "%s -> %s"%(self._to_einops(), term)
        tensor = rearrange(self.tensor, recipe)
        return NamedTensor(tensor, term)

    def _promote(self, dims):
        "Move dims to the front of the line"
        term = " ".join([d for d in self._names if d not in dims]
                        + dims.split()[1:])
        return self._rearrange(term)

    def access(self, dims):
        term = " ".join(dims.split() + [d for d in self._names if d not in dims])
        return self._rearrange(term).tensor




    # def reduce(self, terms, op, **kwargs):
    #     ls = terms.split()
    #     term = " ".join([d for d in self._names
    #                      if d not in ls])
    #     tensor = reduce(self.tensor,
    #                     "%s -> %s"%(self._to_einops(), term), op)
    #     return NamedTensor(tensor, term)

    def op(self, axis_op, dim=None, shift=None):
        kwargs = {}
        if dim is not None:
            assert dim in self._axes, "%s not in %s"%(dim, self._names)
            kwargs["dim"] = self._axes[dim]
        return self._new(axis_op(self.tensor, **kwargs),
                         updates=shift)

    def _force_order(self, names):
        s = ""
        ex = ""
        for d in names:
            if d not in self._names:
                ex += " " + d
                s += " ()"
            else:
                ex += " " + d
                s += " " + d
        tensor = rearrange(self.tensor, "%s -> %s"% (self._to_einops(), s))
        return NamedTensor(tensor, ex)


    def _broadcast_order(self, other):
        order = []
        for d in other._names:
            if d not in self._names:
                order.append(d)
        for d in self._names:
            order.append(d)
        return order

    def _binop(a, op, b):
        order = a._broadcast_order(b)
        a1 = a._force_order(order)
        b1 = b._force_order(order)
        assert_match(a1, b1)
        c = op(a1.tensor, b1.tensor)
        return NamedTensor(c, a1._names)


    def index_select(self, name, index):
        new_names = []
        sizes = []
        for n in self._names:
            if n == name:
                for n2 in index._names:
                    new_names.append(n2)
                    sizes.append(index._sizes[n2])
            else:
                new_names.append(n)
                sizes.append(self._sizes[n])
        return NamedTensor(
            self.tensor.index_select(self._axes[name],
                                     index.tensor.view(-1)).view(*sizes), new_names)

    def narrow(self, change, start, end):
        return self._new(
            self.tensor.narrow(self._axes[change.split("-")[0].strip()], start, end),
            updates=change)

    def softmax(self, name):
        return self._new(F.softmax(self.tensor, dim=self._axes[name]))

    def logsoftmax(self, name):
        return self._new(F.logsoftmax(self.tensor, dim=self._axes[name]))

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
        if methodname in dir(self.tensor):
            method =  getattr(self.tensor, methodname)
            if methodname in _noshift:
                # Call and wrap
                def call(*args, **kwargs):
                    return self._new(method(*args, **kwargs))

            elif methodname in _noshift_dim:
                def call(dim, *args, **kwargs):
                    return self._new(method(self._axes[dim], *args, **kwargs))

            elif methodname in _info:
                # Call and return
                call = method

            elif methodname in _reduce:
                # Call, replace, and wrap
                def call(dim, *args, **kwargs):
                    cur = self
                    method =  getattr(self.tensor, methodname)
                    for d in dim.split():
                        cur = cur._new(method(cur._axes[d], *args, **kwargs), d)
                        method =  getattr(cur.tensor, methodname)
                    return cur
            # elif methodname in _axis_multi:
            #     # Call, replace, and wrap tuple
            #     def call(dim, *args, **kwargs):
            #         dim = self._axes[dim]
            #         results = method(dim, *args, **kwargs)
            #         return tuple((NamedTensor(r, self._names) for r in results))

            elif methodname in _binop:
                def call(other, *args):
                    b = other
                    order = self._broadcast_order(b)
                    a1 = self._force_order(order)
                    b1 = b._force_order(order)
                    method =  getattr(a1.tensor, methodname)
                    assert_match(a1, b1)
                    return a1._new(method(b1.tensor, *args))
            else:
                assert False, "Method not implemented"
            return call
        assert False, "Method does not exist"

def _im_init():
    ## PRINT SETUP
           from PIL.Image import fromarray
           from IPython import get_ipython
           def numpy_to_png(a):
               return fromarray(numpy.array(numpy.clip(a, 0, 1) * 255, 
                                            dtype='uint8'))._repr_png_()
           png = get_ipython().display_formatter.formatters['image/png']
           txt = get_ipython().display_formatter.formatters['text/plain']

           png.for_type(torch.Tensor, lambda t: numpy_to_png(t.numpy()))
           txt.for_type(torch.Tensor, lambda *x: "");
           png.for_type(NamedTensor, lambda t: numpy_to_png(t.tensor.numpy()))
           txt.for_type(NamedTensor, lambda *x: "");
