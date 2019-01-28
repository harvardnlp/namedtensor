import torch.nn as nn

Module = nn.Module
ModuleList = nn.ModuleList


class _Update:
    def rename(self, **kwargs):
        self._updates = kwargs
        return self

    def __call__(self, input):
        updates = {} if "_updates" not in self.__dict__ else self._updates
        return input.op(super(self.__class__, self).forward, **updates)


class _Flat:
    def __call__(self, input):
        return input.op(super(self.__class__, self).forward)


class _Loss:
    def __call__(self, input, target):
        assert self.reduction == "none"
        reduced = input.dims[-1]
        return input.reduce2(
            target, super(self.__class__, self).forward, (reduced,)
        )


class _Augment:
    def augment(self, name):
        self._augment = name
        return self

    def forward(self, input, target):
        augment = (
            "embedding" if "_augment" not in self.__dict__ else self._augment
        )

        return input.augment(super(self.__class__, self).forward, augment)


_wrap = ["Dropout"]


class Dropout(_Flat, nn.Dropout):
    pass


_update = [
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
]


class Linear(_Update, nn.Linear):
    pass


class Conv1d(_Update, nn.Conv1d):
    pass


class Conv2d(_Update, nn.Conv2d):
    pass


class Conv3d(_Update, nn.Conv2d):
    pass


class MaxPool1d(_Update, nn.MaxPool1d):
    pass


class MaxPool2d(_Update, nn.MaxPool2d):
    pass


class MaxPool3d(_Update, nn.MaxPool2d):
    pass


_loss = ["CrossEntropyLoss", "NLLLoss"]


class CrossEntropyLoss(_Loss, nn.CrossEntropyLoss):
    pass


class NLLLoss(_Loss, nn.NLLLoss):
    pass


_augment = ["Embedding"]


class Embedding(_Augment, nn.Embedding):
    pass
