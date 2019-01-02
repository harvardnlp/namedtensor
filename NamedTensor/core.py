from einops import reduce, rearrange
from collections import OrderedDict
import re
import numpy as np

def contract(names, *tensors):
    args = []
    ids = {}
    for t in tensors:
        group = []
        for name in t.dims:
            if name not in ids:
                ids[name] = len(ids)
            group.append(ids[name])
        args.append(t.tensor)
        args.append(group)
    names = names.split()
    args.append([ids[n] for n in names])
    return NamedTensor(np.einsum(*args),
                       names)

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
                batch_dims = [d for d in inp.dims not in spec]
        out = fn(*lifted_inputs)
        return NamedTensor(out, batch_dims + out_spec)
    return lifted


def assert_match(*tensors):
    sizes = {}
    failure = False
    for t in tensors:
        for k, v in t.sizes.items():
            if v == 1: continue
            if k in sizes:
                failure = (failure or sizes[k] != v)
            else:
                sizes[k] = v
    assert not failure, " ".join([str(t.sizes) for t in tensors])


class NamedTensor:
    def __init__(self, tensor, names):
        if isinstance(names, str):
            names = names.split()
        self.tensor = tensor
        self.dims = names
        shape = self.tensor.shape
        self.sizes = OrderedDict(((d, shape[i]) for i, d in enumerate(self.dims)))
        self.axes = OrderedDict(((d, i) for i, d in enumerate(self.dims)))

    def _to_einops(self):
        return " ".join(self.dims)

    def shape(self):
        return self.sizes

    def contract(self, names, *others):
        return contract(names, *([self] + others))


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
        for d in self.dims:
            if d not in names:
                s += " " + d
                ex += " " + d
            elif first:
                s += " (" + strnames + ")"
                ex += " " + dim
                first = False

        tensor = rearrange(self.tensor, "%s -> %s"%(self._to_einops(), s))
        print(ex)
        return NamedTensor(tensor, ex)

    def _split(self, splitstr, **kwargs):
        group = re.match(r"(\w+) -> \(([\w+ ?]+)\)", splitstr)
        dim, strnames = group.groups()
        names = strnames.split()
        query = ""
        ex = ""
        for i, d in enumerate(self.dims):
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
        term = " ".join([d for d in self.dims if d not in dims]
                        + dims.split()[1:])
        return self._rearrange(term)

    def reduce(self, terms, op, **kwargs):
        ls = terms.split()
        term = " ".join([d for d in self.dims
                         if d not in ls])
        tensor = reduce(self.tensor,
                        "%s -> %s"%(self._to_einops(), term), op)
        return NamedTensor(tensor, term)

    def apply(self, dim, axis_op):
        return NamedTensor(axis_op(self.tensor, dim=self.axes[dim]),
                           self.dims)


    def _broadcast_to(self, other):
        s = ""
        ex = ""
        for d in other.dims:
            ex += " " + d
            if d in self.sizes:
                s += " " + d
            else:
                s += " ()"
        for d in self.dims:
            if d not in other.sizes:
                s += " " + d
                ex += " " + d
        tensor = rearrange(self.tensor, "%s -> %s"% (self._to_einops(), s))
        return NamedTensor(tensor, ex)

    def binop(a, op, b):
        a1 = a._broadcast_to(b)
        b1 = b._broadcast_to(a1)
        assert_match(a1, b1)
        c = op(a1.tensor, b1.tensor)
        return NamedTensor(c, a1.dims)


    def __repr__(self):
        return self.tensor
