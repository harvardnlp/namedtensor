import torchtext
from namedtensor import NamedTensor


class NamedField(torchtext.data.Field):
    def __init__(self, **kwargs):
        self.names = kwargs.get("names")
        if self.names is None:
            self.names = ("seqlen",)
        else:
            kwargs.pop("names")
        super(NamedField, self).__init__(**kwargs)

    def numericalize(self, arr, device=None):
        vals = super(NamedField, self).numericalize(arr, device=device)

        if isinstance(vals, list) or isinstance(vals, tuple):
            assert len(vals) == 2
            var, lengths = vals
            if self.sequential and not self.batch_first:
                var = NamedTensor(var, self.names + ("batch",))
            else:
                var = NamedTensor(var, ("batch",) + self.names)
                lengths = NamedTensor(lengths, ("batch",))
            return var, lengths

        else:
            if self.sequential and not self.batch_first:
                var = NamedTensor(vals, self.names + ("batch",))
            else:
                var = NamedTensor(vals, ("batch",) + self.names)
            return var
