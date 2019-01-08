# NamedTensor

## Introduction

A proposal for a named tensor for PyTorch described in the blog post:

http://nlp.seas.harvard.edu/NamedTensor

Currently the library targets the PyTorch ecosystem and Python >= 3.6.

## Usage

```
from namedtensor import ntorch
```

Creation and manipulation:

```
x = ntorch.randn(dict(batch=10, h=10, w=20))
x = x.log()
x = x.float()
x = ntorch.exp(x)
x.shape
```

Transposition:

```
x = x.transpose("batch", "w", "h")

# or early dim stay in place 

x = x.transpose("w", "h")
```

View replacements:

```
x = x.stack(stacked = ("w", "h"))

# Roundtrip

x = x.split(stacked = ("w", "h"), w=20)
```

Dim replacements:

```
x = x.narrow("w", 0, 10)
x = x.softmax("w")
```

Reduction:

```
x = x.mean("w")
x, argmax = x.max("w")
```

Matrix Operations / EinSum:

```
x = ntorch.randn(dict(batch=10, h=10, w=20))
y = ntorch.randn(dict(batch=10, w=20, c=30))
x.dot(y, "w")
```

Lifting Torch Functions

```
linear = nn.Linear(20, 25)
x = x.op(linear)

# or 

x = x.op(linear, wout = "w")

```

## Other Goodies

* Named Distributions libary

## Documentation

http://nlp.seas.harvard.edu/namedtensor/

## Authors

* Alexander Rush (srush@seas.harvard.edu, @harvardnlp)
