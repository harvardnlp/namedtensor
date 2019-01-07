Named Tensor
=============

.. autoclass:: namedtensor.NamedTensor
         :inherited-members:
         :members:


Basic Methods
-------------

These methods return a named tensor of the same form as the original.

         .. method:: _basic(*args)


.. jinja:: tensor

      {% for k in noshift_methods %} :py:meth:`torch.Tensor.{{k}}`  {% endfor %}


Reduction Methods
-----------------

These methods return a named tensor with one or more reduced dimensions

         .. method:: _reduction(dims, *args)


.. jinja:: tensor

      {% for k in reduce_methods %} :py:meth:`torch.Tensor.{{k}}`  {% endfor %}


Tupled Reduction Methods
-------------------------

These methods return a tuple of named tensor with one or more reduced dimensions

         .. method:: _tuple_reduction(dims, *args)


.. jinja:: tensor

      {% for k in multi_reduce_methods %} :py:meth:`torch.Tensor.{{k}}`  {% endfor %}



Non-Tensor Methods
-------------------

These methods return non-tensor information.

         .. method:: _info(*args)


.. jinja:: tensor

      {% for k in info_methods %} :py:meth:`torch.Tensor.{{k}}`  {% endfor %}


Broadcast Methods
-----------------

These methods apply broadcasting before operating between two tensors.

         .. method:: _operate(other, *args)


.. jinja:: tensor

      {% for k in binop_methods %} :py:meth:`torch.Tensor.{{k}}`  {% endfor %}



Named Torch
=============

Named torch `ntorch` is a module that wraps the core torch operations
with named variants. It contains named variants of most of the the
core torch functionality.


Dictionary Builders
----------------------

These methods construct a new named tensor where the sizes are specified
through an ordered dict of names.

.. function:: _build(ordereddict, *args)


.. jinja:: ntorch

      {% for k in build %} :py:func:`torch.{{k}}`  {% endfor %}


Other Builders
----------------

These methods construct a new named tensor where the sizes are specified
through an ordered dict of names.

.. function:: _build(ordereddict, *args)


.. jinja:: ntorch

      {% for k in build %} :py:func:`torch.{{k}}`  {% endfor %}



Basic Functions
----------------

These functions return a named tensor of the same form as the original.

         .. method:: _basic(*args)


.. jinja:: ntorch

      {% for k in noshift %} :py:func:`torch.{{k}}`  {% endfor %}



Distributions
===============

A wrapping of the torch distributions library to make it more clear
to sample and batch the object.



Builders
----------------------

These methods construct a new named distributinon where the sizes are specified
through an ordered dict of names.


.. function:: _build(ordereddict, *args)


.. jinja:: ndistributions

      {% for k in build %} :py:class:`torch.distributions.{{k}}`  {% endfor %}
