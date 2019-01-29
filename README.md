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
x = ntorch.randn(dict(batch=10, h=10, w=20))
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
x = x.stack(stacked = ("w", "h"))

# Roundtrip

x = x.split(stacked = ("w", "h"), w=20)
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

x = ntorch.randn(dict(batch=10, h=10, w=20))
y = ntorch.randn(dict(batch=10, w=20, c=30))
x.dot(y, "w")
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
* Franciso Rivera
* Jiafeng Chen 
