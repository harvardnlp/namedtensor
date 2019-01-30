# NamedTensor

## Introduction

A proposal for a named tensor for PyTorch described in the blog post:

http://nlp.seas.harvard.edu/NamedTensor

Currently the library targets the PyTorch ecosystem and Python >= 3.6.

## Usage

```python
from namedtensor import ntorch
```

Creation and manipulation:

```python
x = ntorch.randn(10, 10, 20, names=("batch", "h", "w"))
x = x.log()
x = x.float()
x = ntorch.exp(x)
x.shape
```

Transposition:

```python
x = x.transpose("batch", "w", "h")

# or early dim stay in place

x = x.transpose("w", "h")
```

View replacements:

```python
x = x.stack(("w", "h"), "stackdim")

# Roundtrip

x = x.split("stackdim", ("w", "h"), w=20)
```

Dim replacements:

```python
x = x.narrow("w", 0, 10)
x = x.softmax("w")
```

Reduction:

```python
x = x.mean("w")
x, argmax = x.max("w")
```

Matrix Operations / EinSum:

```python

x = ntorch.randn(10, 10, 20, names=("batch", "h", "w"))
y = ntorch.randn(10, 20, 30, names=("batch", "w", "c"))
x.dot("w", y)
```

NN Modules

```python

linear = ntorch.nn.Linear(20, 25)
x = linear(x)

# or

linear.rename(wout="w")
x = linear(x)

```

## Other Goodies
* Named NN
* Named Distributions libary

## Documentation

http://nlp.seas.harvard.edu/namedtensor/

## Author

* Alexander Rush (srush@seas.harvard.edu, @harvardnlp)

## Contributors

* Yuntian Deng
* Francisco Rivera
* Jiafeng Chen
* Celine Liang
* Miro Furtado
