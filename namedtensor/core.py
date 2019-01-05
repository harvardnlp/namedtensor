from .schema import _Schema
from einops import rearrange
from collections import OrderedDict
import re


def assert_match(*tensors):
    sizes = {}
    failure = False
    for t in tensors:
        shape = t.shape
        for i, k in t._schema.enum_all():
            v = shape[i]
            if v == 1:
                continue
            if k in sizes:
                failure = failure or sizes[k] != v
            else:
                sizes[k] = v
    assert not failure, " ".join([str(t._sizes) for t in tensors])


class NamedTensorCore:
    def __init__(self, tensor, names, mask=0):
        self._tensor = tensor
        self._schema = _Schema.build(names, mask)

    def _new(self, tensor, drop=None, updates=None, mask=None):
        update_dict = {}
        if updates is not None:
            for u in updates:
                group = re.match(r"(\w+) -> (\w+)", updates)
                start, end = group.groups()
                update_dict[start] = end

        return self.__class__(
            tensor,
            self._schema.drop(drop).update(update_dict),
            self._schema._masked if mask is None else mask,
        )

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
        return OrderedDict(
            ((d, self.shape[i]) for i, d in self._schema.enum_masked())
        )

    def mask_to(self, name):
        if name == "":
            return self._new(self._tensor, mask=0)
        else:
            return self._new(self._tensor, mask=self._schema.get(name) + 1)

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

        tensor = rearrange(
            self._tensor, "%s -> %s" % (self._schema._to_einops(), s)
        )
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

        tensor = rearrange(
            self._tensor,
            "%s -> %s" % (query, ex),
            **{d: kwargs[d] for d in names if d in kwargs}
        )
        return self.__class__(tensor, ex)

    def _rearrange(self, term):
        assert ")" not in term
        recipe = "%s -> %s" % (self._to_einops(), term)
        tensor = rearrange(self._tensor, recipe)
        return self.__class__(tensor, term)

    def _promote(self, dims):
        "Move dims to the front of the line"
        term = " ".join(
            [d for d in self._schema._names if d not in dims]
            + dims.split()[1:]
        )
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
        tensor = rearrange(self._tensor, "%s -> %s" % (self._to_einops(), s))
        return self.__class__(tensor, ex)

    def _broadcast_order(self, other):
        order = []
        for d in other._schema._names:
            if d not in self._schema._names:
                order.append(d)
        for d in self._schema._names:
            order.append(d)
        return order
