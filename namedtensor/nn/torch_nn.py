import torch.nn as nn
from ..torch_helpers import NamedTensor


class Module(nn.Module):
    def register_parameter(self, name, tensor):
        if isinstance(tensor, NamedTensor):
            param = nn.Parameter(tensor.values)
            super(Module, self).register_parameter(
                "_" + name + "_named", param
            )
            tensor._tensor = param
            setattr(self, name, tensor)
        else:
            super(Module, self).register_parameter(name, tensor)

class ModuleList(nn.ModuleList):
    def spec(self, *args, **kwargs):
        for x in self:
            x.spec(*args, **kwargs)
        return self


class _Update:
    def rename(self, **kwargs):
        self._updates = kwargs
        return self

    def __call__(self, input):
        from ..torch_base import ntorch

        if "_spec" in self.__dict__:

            drop = False
            unsplit_dim = False
            if (
                input.values.dim() + 1
                == len(self._input_order) + self._front_pad
            ):
                input = ntorch.stack([input], "tmpdim")
                drop = True
            elif (
                input.values.dim() + 2
                == len(self._input_order) + self._front_pad
            ):
                input = ntorch.stack([input], "tmpdim")
                input = ntorch.stack([input], "tmpdim2")
                drop = True

            elif input.values.dim() > len(self._input_order) + self._front_pad:
                extra = [d for d in input.dims if d not in self._input_order]
                sizes = {x: input.shape[x] for x in extra}
                input = input.stack(extra, "tmpdim")
                if self._front_pad == 2:
                    input = ntorch.stack([input], "tmpdim2")
                unsplit_dim = True

            input = input.transpose(*self._input_order).contiguous()
            updates = {k: v for (v, k) in self._output_update.items()}
            out = input.op(super(_Update, self).forward, **updates)

            if drop:
                out = out.squeeze("tmpdim")

            elif unsplit_dim:
                out = out.split("tmpdim", extra, **sizes)
            if "tmpdim2" in out.shape:
                out = out.squeeze("tmpdim2")
            return out
        else:
            updates = {} if "_updates" not in self.__dict__ else self._updates
            return input.op(super(_Update, self).forward, **updates)


class _Flat:
    def __call__(self, input):
        return input.op(super(_Flat, self).forward)


class _Loss:
    def reduce(self, dims):
        self._reduced = dims
        return self

    def __call__(self, input, target):
        if "_spec" in self.__dict__:
            # Argument of function is batch x target x n1 x ... x nk

            # First figure out batch

            # if self.reduction != "none":
            #     # All other dimensions get

            # else:
            #     # Stack and unstack

            to_batch = [d for d in input.dims if d not in self._input_order]
            sizes = {d: input.shape[d] for d in to_batch}
            input = input.stack(to_batch, "tmpdim")
            target = target.transpose(*to_batch).stack(to_batch, "tmpdim")
            order = ["tmpdim"] + list(self._input_order)
            target_order = ["tmpdim"]

            if self.reduction != "none":
                reduced = ["tmpdim"] + list(self._input_order)
            else:
                reduced = list(self._input_order)

            input = input.transpose(*order).contiguous()
            target = target.transpose(*target_order)
            out = input.reduce2(target, super(_Loss, self).forward, reduced)
            if self.reduction == "none":
                out = out.split("tmpdim", to_batch, **sizes)
            return out

        else:
            assert "_reduced" in dir(
                self
            ), "Call 'spec' with target dimension."
            return input.reduce2(
                target, super(_Loss, self).forward, self._reduced
            )


class _Augment:
    def augment(self, name):
        self._augment = name
        return self

    def forward(self, input):
        if "_spec" in self.__dict__:
            input = input.transpose(*self._input_order).contiguous()
            return input.augment(
                super(_Augment, self).forward, self._output_augment
            )
        else:
            augment = (
                "embedding"
                if "_augment" not in self.__dict__
                else self._augment
            )
            return input.augment(super(_Augment, self).forward, augment)


_wrap = ["Dropout"]


class Dropout(_Flat, nn.Dropout):
    pass


Dropout.__doc__ = nn.Dropout.__doc__


class Linear(_Update, nn.Linear):
    def spec(self, dim_in, name_out=None):
        self._spec = True
        self._front_pad = 0
        self._input_order = (dim_in,)
        self._output_update = {dim_in: name_out if name_out else dim_in}
        return self


class Conv1d(_Update, nn.Conv1d):
    def spec(self, dim_in, dim_conv, name_out=None):
        self._spec = True
        self._front_pad = 1
        self._input_order = (dim_in, dim_conv)
        self._output_update = {dim_in: name_out if name_out else dim_in}
        return self


class Conv2d(_Update, nn.Conv2d):
    def spec(self, dim_in, dims_conv, name_out=None):
        self._spec = True
        self._front_pad = 1
        self._input_order = (dim_in,) + dims_conv
        self._output_update = {dim_in: name_out if name_out else dim_in}
        return self


class Conv3d(_Update, nn.Conv3d):
    def spec(self, dim_in, dims_conv, name_out=None):
        self._spec = True
        self._front_pad = 1
        self._input_order = (dim_in,) + dims_conv
        self._output_update = {dim_in: name_out if name_out else dim_in}
        return self


class MaxPool1d(_Update, nn.MaxPool1d):
    def spec(self, dim_conv):
        self._spec = True
        self._front_pad = 2
        self._input_order = (dim_conv,)
        self._output_update = {}
        return self


class MaxPool2d(_Update, nn.MaxPool2d):
    def spec(self, dims_conv):
        self._spec = True
        self._front_pad = 2
        self._input_order = dims_conv
        self._output_update = {}
        return self


class MaxPool3d(_Update, nn.MaxPool3d):
    def spec(self, dims_conv):
        self._spec = True
        self._front_pad = 2
        self._input_order = dims_conv
        self._output_update = {}
        return self


class ConstantPad1d(_Update, nn.ConstantPad1d):
    def spec(self, dim_pad):
        self._spec = True
        self._front_pad = 2
        self._input_order = (dim_pad,)
        self._output_update = {}
        return self


class ConstantPad2d(_Update, nn.ConstantPad2d):
    def spec(self, dims_pad):
        self._spec = True
        self._front_pad = 2
        self._input_order = dims_pad
        self._output_update = {}
        return self


class ConstantPad3d(_Update, nn.ConstantPad3d):
    def spec(self, dims_pad):
        self._spec = True
        self._front_pad = 2
        self._input_order = dims_pad
        self._output_update = {}
        return self


_update = [
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
]


Linear.__doc__ = nn.Linear.__doc__
Conv1d.__doc__ = nn.Conv1d.__doc__
Conv2d.__doc__ = nn.Conv2d.__doc__
Conv3d.__doc__ = nn.Conv3d.__doc__
MaxPool1d.__doc__ = nn.MaxPool1d.__doc__
MaxPool2d.__doc__ = nn.MaxPool2d.__doc__
MaxPool3d.__doc__ = nn.MaxPool3d.__doc__
ConstantPad1d.__doc__ = nn.ConstantPad1d.__doc__
ConstantPad2d.__doc__ = nn.ConstantPad2d.__doc__
ConstantPad3d.__doc__ = nn.ConstantPad3d.__doc__


class CrossEntropyLoss(_Loss, nn.CrossEntropyLoss):
    def spec(self, dim_target):
        self._spec = True
        self._input_order = (dim_target,)
        return self


class NLLLoss(_Loss, nn.NLLLoss):
    def spec(self, dim_target):
        self._spec = True
        self._input_order = (dim_target,)
        return self


_loss = ["CrossEntropyLoss", "NLLLoss"]

CrossEntropyLoss.__doc__ = nn.CrossEntropyLoss.__doc__
NLLLoss.__doc__ = nn.NLLLoss.__doc__


class Embedding(_Augment, nn.Embedding):
    def spec(self, dim_index, name_embedding):
        self._spec = True
        self._input_order = (dim_index,)
        self._output_augment = name_embedding
        return self


_augment = ["Embedding"]
Embedding.__doc__ = nn.Embedding.__doc__


class _RNN:
    def __call__(self, input, state=None):
        input = input.transpose(*self._input_order).contiguous()

        def run(v, fn):
            if v is None:
                return None
            elif isinstance(v, tuple):
                return tuple((fn(s) for s in v))
            else:
                return fn(v)

        # For some reason, even with batch_first pytorch returns
        # the state with batch second. Need to transpose it.
        state_value = run(
            state, lambda x: x.values.transpose(0, 1).contiguous()
        )
        output, state = super(_RNN, self).forward(input.values, state_value)
        state = run(state, lambda x: x.transpose(0, 1).contiguous())

        updates = self._output_update
        updates2 = dict(updates)
        updates2[self._input_order[0]] = self._layer_name

        # For some reason, even with batch_first pytorch returns
        # the state with batch second.  Need to transpose it.
        state_ret = run(state, lambda x: input._new(x, updates=updates2))
        return input._new(output, updates=self._output_update), state_ret


class RNN(_RNN, nn.RNN):
    def spec(self, dim_in, dim_seq_len, name_out=None, dim_layers="layers"):
        self._layer_name = dim_layers
        self.batch_first = True
        self._spec = True
        self._input_order = (dim_seq_len, dim_in)
        self._name_out = name_out if name_out else dim_in
        self._output_update = {dim_in: name_out if name_out else dim_in}
        return self


class LSTM(_RNN, nn.LSTM):
    def spec(self, dim_in, dim_seq_len, name_out=None, dim_layers="layers"):
        self._layer_name = dim_layers
        self.batch_first = True
        self._spec = True
        self._input_order = (dim_seq_len, dim_in)
        self._name_out = name_out if name_out else dim_in
        self._output_update = {dim_in: name_out if name_out else dim_in}
        return self
