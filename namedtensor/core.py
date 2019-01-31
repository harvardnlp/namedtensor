from .schema import _Schema
from einops import rearrange


def assert_match(*tensors):
    sizes = {}
    failure = False
    for t in tensors:
        shape = t.vshape
        for i, k in t._schema.enum_all():
            v = shape[i]
            if v == 1:
                continue
            if k in sizes:
                failure = failure or sizes[k] != v
            else:
                sizes[k] = v
    assert not failure, " ".join([str(t._sizes) for t in tensors])


class NamedTensorBase:
    """
    Attributes:
        tensor: The raw tensor data
        dims: Tuple of dimension names associated with this array.
        ndim: Number of dimensions
        sizes: The raw dimension sizes
        shape: Ordered mapping from dimension names to lengths.
    """

    def __init__(self, tensor, names, mask=0):
        self._tensor = tensor
        self._schema = _Schema.build(names, mask)
        assert len(self._tensor.shape) == len(self._schema._names), (
            "Tensor has  %d dim, but only %d names"
            % (len(self._tensor.shape), len(self._schema._names))
        )

    @property
    def dims(self):
        "Return the dim names for the tensor"
        return tuple(self._schema._names)

    @property
    def vshape(self):
        "The raw dim size for the tensor."
        return tuple(self._tensor.size())

    @property
    def shape(self):
        "The ordered dict of available dimensions."
        return self._schema.ordered_dict(self._tensor.size())

    @property
    def type(self):
        "Return the type of the tensor"
        return self._tensor.type()

    def __repr__(self):
        return "NamedTensor(\n\t%s,\n\t%s)" % (
            self._tensor,
            self._schema._names,
        )

    def size(self, dim):
        "Return the raw shape of the tensor"
        i = self._schema.get(dim)
        return self._tensor.size(i)

    def assert_size(self, **kwargs):
        "Return the raw shape of the tensor"
        for dim, v in kwargs.items():
            i = self._schema.get(dim)
            assert self._tensor.size(i) == v, (
                "Size of %s should be %d, got %d"
                % (dim, v, self._tensor.size(i))
            )
        return self

    @property
    def values(self):
        "The raw underlying tensor object."
        return self._tensor

    def _new(self, tensor, drop=None, add=None, updates={}, mask=None):
        return self.__class__(
            tensor,
            self._schema.drop(drop).update(updates)._names
            + (() if not add else add),
            self._schema._masked if mask is None else mask,
        )

    def _to_einops(self):
        return self._schema._to_einops()

    def mask_to(self, name):
        if name == "":
            return self._new(self._tensor, mask=0)
        else:
            return self._new(self._tensor, mask=self._schema.get(name) + 1)

    def stack(self, dims, name):
        "Stack any number of existing dimensions into a single new dimension."
        for dim in dims:
            self._schema.get(dim)
        return self._merge(dims, name)

    def split(self, dim, names, **dim_sizes):
        "Split an of existing dimension into new dimensions."
        return self._split(dim, names, dim_sizes)

    def rename(self, dim, name):
        "Rename a dimension."
        return self._split(dim, (name,), {})

    def transpose(self, *dims):
        "Return a new DataArray object with transposed dimensions."
        for dim in dims:
            self._schema.get(dim)
        to_dims = (
            tuple((d for d in self._schema._names if d not in dims)) + dims
        )
        recipe = "%s -> %s" % (self._to_einops(), " ".join(to_dims))
        tensor = rearrange(self._tensor, recipe)
        return self.__class__(tensor, to_dims)

    # Todo: fix arg names
    def _merge(self, names, dim):
        s = ""
        ex = []
        first = True
        for d in self._schema._names:
            if d not in names:
                s += " " + d
                ex.append(d)
            elif first:
                s += " (" + " ".join(names) + ")"
                ex.append(dim)
                first = False
        tensor = rearrange(
            self._tensor, "%s -> %s" % (self._schema._to_einops(), s)
        )
        return self.__class__(tensor, ex)

    def _split(self, dim, names, size_dict):
        query = ""
        ex = []
        for i, d in self._schema.enum_all():
            if d != dim:
                query += " " + d
                ex.append(d)
            else:
                query += " (" + " ".join(names) + ")"
                ex += names

        tensor = rearrange(
            self._tensor,
            "%s -> %s" % (query, " ".join(ex)),
            **{d: size_dict[d] for d in names if d in size_dict}
        )
        return self.__class__(tensor, ex)

    def _rearrange(self, term):
        assert ")" not in term
        recipe = "%s -> %s" % (self._to_einops(), term)
        tensor = rearrange(self._tensor, recipe)
        return self.__class__(tensor, term)

    def __len__(self):
        return len(self._tensor)

    def _promote(self, dims):
        "Move dims to the front of the line"
        term = " ".join(
            [d for d in self._schema._names if d not in dims]
            + dims.split()[1:]
        )
        return self._rearrange(term)

    def _force_order(self, names):
        s = ""
        ex = []
        for d in names:
            if d not in self._schema._names:
                ex.append(d)
                s += " ()"
            else:
                ex.append(d)
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
