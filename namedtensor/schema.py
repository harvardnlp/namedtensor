from collections import OrderedDict
from .utils import make_tuple

class _Axis(object):
    "A dimension"
    def __init__(self, name):    
        self._set_name(name)

    def conflict(self, *axes):
        for axis in axes:
            if self.name == axis.name and self.abbr != axis.abbr:
                return axis
            #if self.abbr == axis.abbr and self.name != axis.name:
            #    return axis
        return False

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
            return self.name == other.name
        elif isinstance(other, str): 
            return other == str(self) or \
                other == self.name or other == self.abbr 

    def __str__(self):
        if self.abbr == self.name[0]:
            return self.name
        else:
            return self.abbr + ":" + self.name

    def __repr__(self):
        return self.abbr + ":" + self.name




class _Schema:
    "Dimension names and order"

    def __init__(self, names, mask=0):
        self._masked = mask
        self._build_axes(names)

    def _build_axes(self, names):
        #print(names)
        axes = []
        for name in names:
            if not isinstance(name, _Axis):
                name = _Axis(name)
            if name in axes:
                raise RuntimeError("Tensor must have unique dims, " +
                    "dim '{}' is not unique, dims={}".format(name, names) +
                    "(Note: dimension names and dimension abbreviations" +
                    "must both be unique)")
            axes.append(name)
        self.axes = tuple(axes)

    @property
    def _names(self):
        return tuple(str(axis) for axis in self.axes)

    @property
    def _abbrs(self):
        return tuple(axis.abbr for axis in self.axes)

    def _to_einops(self):
        return " ".join(self._abbrs)

    @staticmethod
    def build(names, mask=0):
        if isinstance(names, _Schema):
            return _Schema(names.axes, mask)
        return _Schema(names, mask)

    def get(self, name):
        dim = None
        for i, n in self.enum_all():
            if name == n:
                if i < self._masked:
                    raise RuntimeError("Dimension {} is masked".format(name,))
                if dim is None:
                    dim = i
                else:
                    raise RuntimeError(
                        "Ambiguity in axis name, '{}'' matches '{}', ".\
                            format(name, self.axes[dim]) +
                        "and also '{}'".format(self.axes[i])
                    )
        if dim is not None:
            return dim 
        elif name not in self.axes:
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
        #names = [_Axis(name) for name in make_tuple(names)]
        new_axes = [ n for n in self.axes if n not in make_tuple(names)]
        return _Schema(
            [ str(a) for a in new_axes],
            mask=self._masked )

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
        return enumerate(self.axes[self._masked :], self._masked)

    def enum_all(self):
        return enumerate(self.axes)

    def merge(self, other):
        axes = list(self.axes)
        for a in other.axes:
            if a not in self.axes:
                axes.append(a)
            elif a.conflict(*self.axes):
                raise RuntimeError( 
                    "Axis {} conflicts with axes {}".\
                    format(a, self.axes)
                )
        return self.__class__(axes)


