import torch
from .torch_helpers import build

_build = {"ones", "zeros", "randn"}

_unary = {"abs", "acos",  "asin", "atan",
           "ceil", "cos", "cosh",
           "exp", "expm1", "log", "rsqrt",
            "sigmoid", "sign", "sin", "sinh", "sqrt",
          "tan", "tanh", "tril", "triu"}


_noshift = {"abs", "acos",  "asin", "atan", "byte",
           "ceil", "clamp", "clone", "contiguous",
           "cos", "cosh", "cpu", "cuda", "double",
           "exp", "expm1", "float", "floor", "fmod",
           "frac", "half", "int", "long", "log", "pow",
           "reciprical", "round", "rsqrt", "short",
            "sigmoid", "sign", "sin", "sinh", "sqrt",
            "sub", "to", "tan", "tanh", "tril", "triu",
            "trunc"}



class MyMetaclass(type):
    def __getattr__(cls, name):
        print(name)
        if name in _build:
            def call(names, *args, **kwargs):
                return build(getattr(torch, name), names, *args, **kwargs)
            return call
        elif name in _noshift:
            def call(ntensor, *args, **kwargs):
                return getattr(ntensor, name)(*args, **kwargs)
            return call

class ntorch(metaclass=MyMetaclass):
    pass
