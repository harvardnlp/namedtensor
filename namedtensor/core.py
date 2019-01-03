from .schema import _Schema
from einops import reduce, rearrange
from collections import OrderedDict
import re

# def lift(fn, in_specs, out_spec):
#     in_specs = [s.split() for s in in_specs]
#     out_spec = out_spec.split()
#     def lifted(*inputs):
#         assert_match(inputs)
#         assert len(inputs) == len(in_specs)
#         lifted_inputs = []
#         batch_dims = []
#         for inp, spec in zip(inputs, in_specs):
#             lifted_inputs.append(inp._promote(spec).tensor
#                                  if spec is not None else inp)
#             if spec is not None:
#                 batch_dims = [d for d in inp._names not in spec]
#         out = fn(*lifted_inputs)
#         return NamedTensor(out, batch_dims + out_spec)
#     return lifted


def assert_match(*tensors):
    sizes = {}
    failure = False
    for t in tensors:
        shape = t.shape
        for i, k in t._schema.enum_all():
            v = shape[i]
            if v == 1: continue
            if k in sizes:
                failure = (failure or sizes[k] != v)
            else:
                sizes[k] = v
    assert not failure, " ".join([str(t._sizes) for t in tensors])



class NamedTensorCore:
    def __init__(self, tensor, names, mask=0):
        self._tensor = tensor
        self._schema = _Schema.build(names, mask)

    def _new(self, tensor, drop=None, updates=None):
        update_dict = {}
        if updates is not None:
            for u in updates:
                group = re.match(r"(\w+) -> (\w+)", updates)
                start, end = group.groups()
                update_dict[start] = end
        return self.__class__(tensor, self._schema.drop(drop).update(update_dict))

    @property
    def tensor(self):
        "Return the raw tensor"
        return self._tensor

    @property
    def shape(self):
        "Return the raw shape of the tensor"
        return self._tensor.shape

    @property
    def named_shape(self):
        "Return an ordered dict of the available dimensions"
        return OrderedDict(((d, self.shape[i])
                            for i, d in self._schema.enum_masked()))

    def _size(self, dim):
        i = self._schema.get(dim)
        return self.shape[i]

    def _to_einops(self):
        return self._schema._to_einops()

    def shift(self, *ops, **kwargs):
        """
        A small transposition language for moving around dimensions
        within a named tensor.
        """
        cur = self
        for op in ops:
            if op.strip().startswith("("):
                cur = cur._merge(op)
            elif op.strip().endswith(")"):
                cur = cur._split(op, **kwargs)
            elif op.strip().startswith("..."):
                cur = cur._promote(op)
            else:
                cur = cur._rearrange(op)
        return cur

    def _merge(self, mergestr):
        group = re.match(r"\(([\w+ ?]+)\) -> (\w+)", mergestr)
        shape = self.shape
        strnames, dim = group.groups()
        names = strnames.split()
        s = ""
        ex = ""
        first = True
        for d in self._schema._names:
            if d not in names:
                s += " " + d
                ex += " " + d
            elif first:
                s += " (" + strnames + ")"
                ex += " " + dim
                first = False

        tensor = rearrange(self._tensor, "%s -> %s"%(self._schema._to_einops(), s))
        return self.__class__(tensor, ex)

    def _split(self, splitstr, **kwargs):
        group = re.match(r"(\w+) -> \(([\w+ ?]+)\)", splitstr)
        dim, strnames = group.groups()
        names = strnames.split()
        query = ""
        ex = ""
        for i, d in self._schema.enum_all():
            if d != dim:
                query += " " + d
                ex += " " + d
            else:
                query += " (" + strnames + ")"
                ex += " " + strnames

        tensor = rearrange(self._tensor, "%s -> %s"%(query, ex),
                           **{d:kwargs[d] for d in names
                              if d in kwargs})
        return self.__class__(tensor, ex)

    def _rearrange(self, term):
        assert ")" not in term
        recipe = "%s -> %s"%(self._to_einops(), term)
        tensor = rearrange(self._tensor, recipe)
        return self.__class__(tensor, term)

    def _promote(self, dims):
        "Move dims to the front of the line"
        term = " ".join([d for d in self._schema._names
                         if d not in dims]
                        + dims.split()[1:])
        return self._rearrange(term)

    def _force_order(self, names):
        s = ""
        ex = ""
        for d in names:
            if d not in self._schema._names:
                ex += " " + d
                s += " ()"
            else:
                ex += " " + d
                s += " " + d
        tensor = rearrange(self._tensor, "%s -> %s"% (self._to_einops(), s))
        return self.__class__(tensor, ex)


    def _broadcast_order(self, other):
        order = []
        for d in other._schema._names:
            if d not in self._schema._names:
                order.append(d)
        for d in self._schema._names:
            order.append(d)
        return order

    def _binop(a, op, b):
        order = a._broadcast_order(b)
        a1 = a._force_order(order)
        b1 = b._force_order(order)
        assert_match(a1, b1)
        c = op(a1._tensor, b1._tensor)
        return self.__class__(c, a1._names)
