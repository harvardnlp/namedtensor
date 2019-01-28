import torch.nn as nn


class Linear(nn.Linear):
    def rename(self, **kwargs):
        self._updates = kwargs
        return self

    def forward(self, input):
        updates = {} if "_updates" not in self.__dict__ else self._updates
        return input.op(super(self.__class__, self).forward, **updates)


class Conv1d(nn.Conv1d):
    def rename(self, **kwargs):
        self._updates = kwargs
        return self

    def forward(self, input):
        updates = {} if "_updates" not in self.__dict__ else self._updates
        return input.op(super(self.__class__, self).forward, **updates)


class Conv2d(nn.Conv2d):
    def rename(self, **kwargs):
        self._updates = kwargs
        return self

    def forward(self, input):
        updates = {} if "_updates" not in self.__dict__ else self._updates
        return input.op(super(self.__class__, self).forward, **updates)


class Dropout(nn.Dropout):
    def forward(self, input):
        return input.op(super(self.__class__, self).forward)


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input, target):
        assert self.reduction == "none"
        reduced = input.dims[-1]
        return input.reduce2(
            target, super(self.__class__, self).forward, (reduced,)
        )


class NLLLoss(nn.NLLLoss):
    def forward(self, input, target):
        assert self.reduction == "none"
        reduced = input.dims[-1]
        return input.reduce2(
            target, super(self.__class__, self).forward, (reduced,)
        )
