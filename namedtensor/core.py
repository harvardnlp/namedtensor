from einops import reduce, rearrange
from collections import OrderedDict
import re
import numpy as np
import torch
import opt_einsum as oe

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


class NamedTensor:
    def __init__(self, tensor, names):
        if isinstance(names, str):
            names = names.split()
        self.tensor = tensor
        self._names = names
        shape = self.tensor.shape
        self._sizes = OrderedDict(((d, shape[i]) for i, d in enumerate(self._names)))
        self._axes = OrderedDict(((d, i) for i, d in enumerate(self._names)))

    def _to_einops(self):
        return " ".join(self._names)

    @property
    def named_shape(self):
        return self._sizes

    def contract(self, names, *others):
        return contract(names, *((self,) + others))


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

    def reduce(self, terms, op, **kwargs):
        ls = terms.split()
        term = " ".join([d for d in self._names
                         if d not in ls])
        tensor = reduce(self.tensor,
                        "%s -> %s"%(self._to_einops(), term), op)
        return NamedTensor(tensor, term)

    def apply(self, axis_op, dim=None):
        kwargs = {}
        if dim is not None:
            assert dim in self._axes, "%s not in %s"%(dim, self._names)
            kwargs["dim"] = self._axes[dim]
        return NamedTensor(axis_op(self.tensor, **kwargs), self._names)


    def _broadcast_to(self, other):
        s = ""
        ex = ""
        broadcast = (set(other._names) |  set(self._names)) - (set(other._names) & set(self._names))

        for d in other._names:
            if d in broadcast:
                ex += " " + d
                s += " ()"
        for d in self._names:
            s += " " + d
            ex += " " + d
        tensor = rearrange(self.tensor, "%s -> %s"% (self._to_einops(), s))
        return NamedTensor(tensor, ex)

    def binop(a, op, b):
        a1 = a._broadcast_to(b)
        b1 = b._broadcast_to(a1)
        assert_match(a1, b1)
        c = op(a1.tensor, b1.tensor)
        return NamedTensor(c, a1._names)


    def __add__(self, b):
        return self.binop(torch.add, b)

    def __repr__(self):
        return self.tensor
