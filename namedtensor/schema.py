from collections import OrderedDict
from .utils import make_tuple

class _Axis(object):
    "A dimension"
    def __init__(self, name, size=None):    
        self._set_name(name)
        self.size = size

    def conflict(self, *axes):
        sizes = []
        if self.size != 1 and not self.size is None:
            sizes = [self.size, ]
        for axis in axes:
            if self.name == axis.name and self.abbr != axis.abbr:
                return True
            if self.abbr == axis.abbr and self.name != axis.name:
                return True
            if self.abbr == axis.abbr and self.name == axis.name:
                if axis.size != 1 and not axis.size is None:
                    sizes.append(axis.size)
        return not all(x == sizes[0] for x in sizes)

    def _set_name(self, name):
        names = name.split(':')
        if len(names) == 1:
            self.name = names[0]
            self.abbr = self.name[0]
        elif len(names) == 2:
            self.name = names[1]
            self.abbr = names[0]
            if not len(self.abbr) == 1:
                raise RuntimeError("Error setting axis name {}\n".format(name) +
                    "Abbreviations must be a single character")
        else:
            raise RuntimeError("Error setting axis name {}\n".format(name) +
                "Valid names are of the form 'name' or 'n:name'")

    def __eq__(self, other):
        if isinstance(other, _Axis):
            return self.name == other.name or self.abbr == other.abbr
        elif isinstance(other, str):
            return str(self) == other or self.name==other or self.abbr==other 

    def __str__(self):
        return self.abbr + ":" + self.name

    def __repr__(self):
        return str(self)




class _Schema:
    "Dimension names and order"

    def __init__(self, names, sizes=None, mask=0):
        self._masked = mask
        names= make_tuple(names)
        if sizes is not None and len(sizes) != len(names):
            raise RuntimeError("Error setting schema shape, " +
                "'{}' does not match {}".format(sizes, names))

        axes = []
        for i, name in enumerate(names):
            if not isinstance(name, _Axis):
                name = _Axis(name, sizes[i])
            elif sizes is not None and name.size != sizes[i]:
                raise RuntimeError("Error setting schema dimension, " +
                    "dimension '{}' has size {} but attempting to set with {}".\
                    format(name, name.size, sizes[i]))
            if name in axes:
                raise RuntimeError("Tensor must have unique dims, " +
                    "dim '{}' is not unique, dims={}".format(name, names) +
                    "(Note: dimension names and dimension abbreviations" +
                    "must both be unique)")
            axes.append(name)
        self._axes = tuple(axes)

    @property
    def _names(self):
        return tuple(str(axis) for axis in self._axes)

    @property
    def _abbrs(self):
        return tuple(axis.abbr for axis in self._axes)

    def _to_einops(self):
        return " ".join(self._abbrs)

    def ordered_dict(self):
        return OrderedDict((str(a), a.size) for a in self._axes)

    @staticmethod
    def build(names, sizes, mask=0):
        if isinstance(names, _Schema):
            return _Schema(names._axes, sizes, mask)
        return _Schema(names, sizes, mask)

    def get(self, name):
        for i, n in self.enum_all():
            if name == n:
                if i < self._masked:
                    raise RuntimeError("Dimension {} is masked".format(name,))
                return i
        if name not in self._axes:
            raise RuntimeError(
                "Dimension {} does not exist. Available dimensions are {}".\
                format(name, self._names)
            )
        else:
            raise RuntimeError( # Not sure how we'd get here
                "Something unexpected occured while searching for {} in {}".\
                format(name, self._names)
            )
        
        
    def drop(self, names):
        names = [_Axis(name) for name in make_tuple(names)]
        return _Schema(
            [n for n in self._axes if n not in names], mask=self._masked
        )

    def update(self, update):
        if not update:
            return self
        raise RuntimeError("Update err %s" % update)
    #    fail = True
    #    for n in self._names:
    #        if n in update:
    #            fail = False
    #    if fail:
    #        raise RuntimeError("Tried to update unknown dim %s" % update)
    #    return _Schema([update.get(n, n) for n in self._names], self._masked)

    def enum_masked(self):
        return enumerate(self._axes[self._masked :], self._masked)

    def enum_all(self):
        return enumerate(self._axes)
