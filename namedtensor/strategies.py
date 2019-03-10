from . import ntorch
from hypothesis.strategies import (
    text,
    composite,
    sampled_from,
    lists,
    permutations,
    integers,
    floats,
)
from hypothesis.extra.numpy import arrays, array_shapes
import numpy as np


# Setup Hypothesis helpers
def named_tensor(dtype=np.float32, shape=array_shapes(2, 5, max_side=5)):
    @composite
    def name(draw, array):
        array = draw(array)
        names = draw(
            lists(
                text(min_size=1, alphabet="abc"),
                max_size=len(array.shape),
                min_size=len(array.shape),
                unique=True,
            )
        )
        return ntorch.tensor(array, names=names)

    return name(
        arrays(dtype, shape, elements=floats(min_value=-1e9, max_value=1e9))
    )


def dim(tensor):
    return sampled_from(list(tensor.shape.keys()))


def dims(tensor, min_size=2, max_size=5):
    return lists(
        dim(tensor), unique=True, min_size=min_size, max_size=max_size
    )


def name(tensor):
    return text(alphabet="abc", min_size=1).filter(
        lambda y: y not in tensor.shape
    )


def names(tensor, max_size=5):
    return lists(name(tensor), unique=True, min_size=2, max_size=max_size)


def broadcast_named_tensor(x, dtype=np.float32):
    @composite
    def fill(draw):
        ds = draw(dims(x, max_size=2))
        ns = draw(names(x, max_size=2))
        perm = draw(permutations(range(len(ns) + len(ds))))

        def reorder(ls):
            return [ls[perm[i]] for i in range(len(ls))]

        sizes = draw(
            lists(
                integers(min_value=1, max_value=4),
                min_size=len(ns),
                max_size=len(ns),
            )
        )
        shape = reorder([x.shape[d] for d in ds] + sizes)
        np = draw(arrays(dtype, shape=shape))

        return ntorch.tensor(np, names=reorder(ds + ns))

    return fill()


def mask_named_tensor(x, dtype=np.uint8):
    @composite
    def fill(draw):
        ds = draw(dims(x, max_size=2))
        perm = draw(permutations(range(len(ds))))

        def reorder(ls):
            return [ls[perm[i]] for i in range(len(ls))]

        shape = reorder([x.shape[d] for d in ds])
        np = draw(arrays(dtype, shape, integers(min_value=0, max_value=1)))

        return ntorch.tensor(np, names=reorder(ds)).byte()

    return fill()
