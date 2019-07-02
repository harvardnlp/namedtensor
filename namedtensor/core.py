from .schema import _Schema
import operator
import functools


def prod(factors):
    return functools.reduce(operator.mul, factors, 1)


def assert_match(*tensors):
    sizes = {}
    failure = False
    axes = []
    for tensor in tensors:
        axes = axes + list(tensor.dims)
    for ax in axes:
        assert not ax.conflict(*axes), "Overlapping dim names must match: " + " ".join(
            [str(t.shape) for t in tensors]
        )


class NamedTensorBase:
    """
    Attributes:
        tensor: The raw tensor data
        dims: Tuple of unique dimension names associated with this array.
        ndim: Number of dimensions
        sizes: The raw dimension sizes
        shape: Ordered mapping from dimension names to lengths.
    """

    def __init__(self, tensor, names, mask=0):
        self._tensor = tensor
        self._schema = _Schema.build(names, mask)

    def __deepcopy__(self, memo):
        new_ntensor = self._new(self._tensor.__deepcopy__(memo))
        memo[id(self)] = new_ntensor
        return new_ntensor

    @property
    def dims(self):
        "Return the dim names for the tensor"
        return tuple(self._schema.axes)

    @property
    def vshape(self):
        "The raw dim size for the tensor."
        return tuple(self._tensor.size())

    @property
    def shape(self):
        "The ordered dict of available dimensions."
        return tuple(zip(self.dims, self._tensor.shape))

    def __repr__(self):
        return "NamedTensor(\n\t{},\n\t{})".format(
            self._tensor,
            self.dims,
        )

    def size(self, dim):
        "Return the raw shape of the tensor"
        return self._tensor.size(self._schema.get(dim))

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
        #raise RuntimeError("Err drop=%s" % drop +
        #    " add=%s" % add + "updates=%s" % updates )
        return self.__class__(
            tensor,
            self._schema.drop(drop).update(updates).axes
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
        return self._stack(dims, name)

    def split(self, dim, names, **dim_sizes):
        "Split an of existing dimension into new dimensions."
        return self._split(dim, names, dim_sizes)

    def rename(self, dim, name):
        "Rename a dimension."
        return self._split(dim, (name,), {})

    def transpose(self, *dims):
        "Return a new DataArray object with transposed dimensions."
        to_dims = ( tuple(d for d in self.dims if d not in dims) + dims )
        indices = [ self._schema.get(d) for d in to_dims ]
        tensor = self._tensor.permute(*indices)
        return self.__class__(tensor, to_dims)

    def _stack(self, names, dim):
        trans = []
        new_schema = []
        first = True
        view = []
        for d in self.dims:
            if d not in names:
                trans.append(d)
                new_schema.append(d)
                view.append(self.size(d))
            elif first:
                trans += names
                view.append(prod([self.size(d2) for d2 in names]))
                new_schema.append(dim)
                first = False
        tensor = self.transpose(*trans)._tensor.contiguous().view(*view)
        return self.__class__(tensor, new_schema)

    def _splitdim(self, dim, names, size_dict):
        new_schema = []
        view = []
        for i, d in self._schema.enum_all():
            if d != dim:
                new_schema.append(d)
                view.append(self.size(d))
            else:
                for d2 in names:
                    d2 = d2.split(":")
                    view.append(size_dict.get(d2[-1], size_dict.get(d2[0],-1)))
                new_schema += names
        return self.__class__(self._tensor.view(*view), new_schema)


    def __len__(self):
        return len(self._tensor)

    def _promote(self, dims):
        """ Move dims to the front of the line """
        raise RuntimeError("Err %s" % dims)

        term = [
            d for d in self.dims if d not in dims
        ] + dims.split()[1:]

        return self.transpose(*term)

    def _force_order(self, schema):
        """ Forces self to take order in names, adds 1-size dims if needed """
        new_schema = []
        view = []
        trans = []
        for d in schema.axes:
            if d not in self.dims:
                new_schema.append(d)
                view.append(1)
            else:
                new_schema.append(d)
                view.append(self.size(d))
                trans.append(d)
        return self.__class__(
            self.transpose(*trans)._tensor.contiguous().view(*view), new_schema
        )

    def _broadcast_order(self, other):
        """ Outputs a shared order (list) that works for self and other """
        return self._schema.merge(other._schema)
        #order = []
        #for d in other_names:
        #    if d not in self.dims:
        #        order.append(d)
        #for d in self.dims:
        #    order.append(d)
        #return order

    def _mask_broadcast_order(self, main_names):
        """
        If broadcasting possible from self (mask) to main, outputs a shared order.
        Otherwise errors and prints dimensions that exist in mask but not in main.
        """
        raise RuntimeError("Err %s" % main_names)
        to_be_broadcasted = set(self.dims)
        broadcasted_to = set(main_names)

        diff = to_be_broadcasted.difference(broadcasted_to)
        diff_string = ", ".join(diff)

        assert len(diff) == 0, (
            "Attemped to broadcast mask but unable to broadcast dimensions %s"
            % diff_string
        )

        return main_names
