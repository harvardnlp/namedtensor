import torchtext
import torch
from collections import namedtuple

from namedtensor import NamedTensor



JaggedInfo = namedtuple("JaggedInfo", ["lengths", "mask"])


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

            # Compute the mask based on lengths
            idxs = torch.arange(0, max(lengths)).to(lengths)
            mask = idxs.repeat(len(lengths), 1) >= lengths.unsqueeze(-1)
            mask = NamedTensor(mask, names=("batch",) + self.names)

            # Convert lengths to NamedTensor
            lengths = NamedTensor(lengths, ("batch",))

            # Construct a JaggedInfo
            jagged_info = JaggedInfo(lengths = lengths, mask = mask)

            return var, jagged_info

        else:
            if self.sequential and not self.batch_first:
                var = NamedTensor(vals, self.names + ("batch",))
            else:
                var = NamedTensor(vals, ("batch",) + self.names)
            return var
