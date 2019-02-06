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


ModuleList = nn.ModuleList


class _Update:
    def rename(self, **kwargs):
        self._updates = kwargs
        return self

    def __call__(self, input):
        if "_spec" in self.__dict__:
            input = input.transpose(*self._input_order).contiguous()
            updates = {k: v for (v, k) in self._output_update.items()}
            return input.op(super(_Update, self).forward, **updates)
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
            reduced = list(self._input_order)
            if self.reduction != "none":
                reduced = input._schema._names
            input = input.transpose(*self._input_order).contiguous()
            return input.reduce2(target, super(_Loss, self).forward, reduced)
        else:
            assert "_reduced" in dir(self), "Call 'spec' with target dimension."
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
        self._input_order = (dim_in,)
        self._output_update = {dim_in: name_out if name_out else dim_in}
        return self


class Conv1d(_Update, nn.Conv1d):
    def spec(self, dim_in, dim_conv, name_out=None):
        self._spec = True
        self._input_order = (dim_in, dim_conv)
        self._output_update = {dim_in: name_out if name_out else dim_in}
        return self


class Conv2d(_Update, nn.Conv2d):
    def spec(self, dim_in, dims_conv, name_out=None):
        self._spec = True
        self._input_order = (dim_in,) + dims_conv
        self._output_update = {dim_in: name_out if name_out else dim_in}
        return self


class Conv3d(_Update, nn.Conv2d):
    def spec(self, dim_in, dims_conv, name_out=None):
        self._spec = True
        self._input_order = (dim_in,) + dims_conv
        self._output_update = {dim_in: name_out if name_out else dim_in}
        return self


class MaxPool1d(_Update, nn.MaxPool1d):
    def spec(self, dim_in, dim_conv, name_out=None):
        self._spec = True
        self._input_order = (dim_in, dim_conv)
        self._output_update = {dim_in: name_out if name_out else dim_in}
        return self


class MaxPool2d(_Update, nn.MaxPool2d):
    def spec(self, dim_in, dims_conv, name_out=None):
        self._spec = True
        self._input_order = (dim_in,) + dims_conv
        self._output_update = {dim_in: name_out if name_out else dim_in}
        return self


class MaxPool3d(_Update, nn.MaxPool2d):
    def spec(self, dim_in, dims_conv, name_out=None):
        self._spec = True
        self._input_order = (dim_in,) + dims_conv
        self._output_update = {dim_in: name_out if name_out else dim_in}
        return self


_update = [
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
]


Linear.__doc__ = nn.Linear.__doc__
Conv1d.__doc__ = nn.Conv1d.__doc__
Conv2d.__doc__ = nn.Conv2d.__doc__
Conv3d.__doc__ = nn.Conv3d.__doc__
MaxPool1d.__doc__ = nn.MaxPool1d.__doc__
MaxPool2d.__doc__ = nn.MaxPool2d.__doc__
MaxPool3d.__doc__ = nn.MaxPool3d.__doc__


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

        if state is None:
            state_value = None
        elif isinstance(state, tuple):
            state_value = tuple((s.values.transpose(0, 1) for s in state))
        else:
            state_value = state.values

        output, state = super(_RNN, self).forward(input.values,
                                                  state_value)
        if isinstance(state, tuple):
            state = tuple((s.transpose(0, 1) for s in state))
        else:
            state = state.transpose(0, 1)
        updates = self._output_update
        updates2 = dict(updates)
        updates2[self._input_order[0]] = self._layer_name

        if isinstance(state, tuple):
            state_ret = tuple((input._new(s, updates=updates2) for s in state))
        else:
            state_ret = input._new(state, updates=updates2)
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
