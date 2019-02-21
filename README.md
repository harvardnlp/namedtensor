# NamedTensor

[![Build Status](https://travis-ci.org/harvardnlp/namedtensor.svg?branch=master)](https://travis-ci.org/harvardnlp/namedtensor) [![Coverage Status](https://coveralls.io/repos/github/harvardnlp/namedtensor/badge.svg?branch=master)](https://coveralls.io/github/harvardnlp/namedtensor?branch=master)

## Introduction

A proposal for a named tensor for Torch described here:

http://nlp.seas.harvard.edu/NamedTensor


NamedTensor is an thin-wrapper on Torch tensor that makes three changes to the API:

1) **Naming**: Dimension access and reduction use a named `dim` argument instead of an index. Constructing and adding dimensions use a `name` argument. Axis-based indexing \[ \] is replaced by named indexing. 
2) **Broadcasting**: All functions broadcast based on set-operations not through heuristic ordering rules, e.g. if `z = x + y` then `z` has the union of the dimension in `x` and `y`.    
3) **Lifting**: Order-based functions can be lifted by providing name annotations through `.spec` methods. For instance, convolution requires the user to name the channel and kernel dims, e.g .`conv2d.spec("channel", ("x", "y"))`. This provides dynamic checks, better error messages, and consistent documentation.

## Usage

```python
from namedtensor import ntorch
```

### Building tensors.

All pytorch builders have an extra keyword argument names.

```python
x = ntorch.randn(10, 10, 20, names=("batch", "h", "w"))
x = ntorch.ones(10, 10, 20, names=("batch", "h", "w"))
```

### Standard functions

All functions that keep dimensionality work in the same way.

```python
x = x.log()
x = x.float()
x = ntorch.exp(x)
```

### Named Indexing

Indexing and masking operation work by name as opposed to absolute position. 

```python
first_batch = x[{"batch": 1}]
three_examples = x[{"batch": slice(1, 4)}]
masked = x[ x > 0.5 ]
```

Advanced indexing by named tensors.

```python
select = ntorch.tensor([1, 4, 5], names=("rows",))
y = x[{"h": select}] 
# y shape ("batch", "rows", "w")
```

### No view or unsqueeze

View, tranpose, and friends are deprecated in favor of named
access and movement.

```python
x = x.stack(("w", "h"), "stackdim")

# Roundtrip

x = x.split("stackdim", ("w", "h"), w=20)
```

There is no need to ever have unsqueeze since broadcasting is done by name overlap. 

Similar notation can be used for setting values.


### All methods take named args

Any function with a `dim` argument now can be accessed based on the
dimension name.

```python
x = x.narrow("w", 0, 10)
x = x.softmax("w")
```

This is true of reductions functions as well, where the named
dimension is eliminated.

```python
x = x.mean("w")
x, argmax = x.max("w")
```

### Tensor contractions

Matrix operations also use the dimension arguments.
We can replace einsum based on persistent names.

```python

x = ntorch.randn(10, 10, 20, names=("batch", "h", "w"))
y = ntorch.randn(10, 20, 30, names=("batch", "w", "c"))
x.dot("w", y)
```

This also makes indexing much easier to read.

```python

x = ntorch.ones(10, 10, 20, names=("batch", "time", "vocab"))
y = ntorch.randn(20, 30, names=("vocab", "embsize"))
y.index_select("vocab", x)
```

## NN Modules

NN units no longer take ordered tensors. They now have a required additional method `spec`
that lets the user set the the input and output dimensions of the object. 

Examples

```python
  conv = ntorch.nn.Conv1d(5, 10, 2).spec("input", "time", "output")
  n = ntorch.randn(20, 30, 5, names=("batch", "time", "input"))
  out = conv(n)
```

```python
  drop = ntorch.nn.Dropout()
  n = ntorch.randn(4, 20, names=("batch", "target"))
  out = drop(n)
```

```python
  loss = ntorch.nn.NLLLoss().spec("target")
  predict = ntorch.randn(20, 4, names=("target", "batch"))
  target = ntorch.tensor([2, 2, 3, 4], ["batch"])
  out = loss(predict, target)
```

## Other Goodies

* Named Distributions libary

## Documentation

http://nlp.seas.harvard.edu/namedtensor/

## Author

* Alexander Rush (srush@seas.harvard.edu, @harvardnlp)

## Contributors

(NamedTensor is being collectively developed by Harvard CS 287)

* Yuntian Deng
* Justin Chiu
* Francisco Rivera
* Jiafeng Chen
* Celine Liang
* Miro Furtado
* Roshan Padaki
* Mirac Suzgun
* Belén Saldías
* Jason Ren
* Josh Feldman
* Jambay Kinley
