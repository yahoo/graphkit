Operations
==========

At a high level, an operation is a node in a computation graph.  GraphKit uses an ``operation`` class to represent these computations.

The ``operation`` class
-----------------------

The ``operation`` class specifies an operation in a computation graph, including its input data dependencies as well as the output data it provides.  It provides a lightweight wrapper around an arbitrary function to make these specifications.

There are many ways to instantiate an ``operation``, and we'll get into more detail on these later.  First off, though, here's the specification for the ``operation`` class:

.. autoclass:: graphkit.operation
   :members: __init__, __call__
   :member-order: bysource


Operations are just functions
------------------------------

At the heart of each ``operation`` is just a function, any arbitrary function.  Indeed, you can instantiate an ``operation`` with a function and then call it just like the original function, e.g.::

   >>> from operator import add
   >>> from graphkit import operation

   >>> add_op = operation(name='add_op', needs=['a', 'b'], provides=['a_plus_b'])(add)

   >>> add_op(3, 4) == add(3, 4)
   True


Specifying graph structure: ``provides`` and ``needs``
------------------------------------------------------

Of course, each ``operation`` is more than just a function.  It is a node in a computation graph, depending on other nodes in the graph for input data and supplying output data that may be used by other nodes in the graph (or as a graph output).  This graph structure is specified via the ``provides`` and ``needs`` arguments to the ``operation`` constructor.  Specifically:

* ``provides``: this argument names the outputs (i.e. the returned values) of a given ``operation``.  If multiple outputs are specified by ``provides``, then the return value of the function comprising the ``operation`` must return an iterable.

* ``needs``: this argument names data that is needed as input by a given ``operation``.  Each piece of data named in needs may either be provided by another ``operation`` in the same graph (i.e. specified in the ``provides`` argument of that ``operation``), or it may be specified as a named input to a graph computation (more on graph computations :ref:`here <graph-computations>`).

When many operations are composed into a computation graph (see :ref:`graph-composition` for more on that), Graphkit matches up the values in their ``needs`` and ``provides`` to form the edges of that graph.

Let's look again at the operations from the script in :ref:`quick-start`, for example::

   >>> from operator import mul, sub
   >>> from graphkit import compose, operation

   >>> # Computes |a|^p.
   >>> def abspow(a, p):
   ...   c = abs(a) ** p
   ...   return c

   >>> # Compose the mul, sub, and abspow operations into a computation graph.
   >>> graphop = compose(name="graphop")(
   ...    operation(name="mul1", needs=["a", "b"], provides=["ab"])(mul),
   ...    operation(name="sub1", needs=["a", "ab"], provides=["a_minus_ab"])(sub),
   ...    operation(name="abspow1", needs=["a_minus_ab"], provides=["abs_a_minus_ab_cubed"], params={"p": 3})(abspow)
   ... )

The ``needs`` and ``provides`` arguments to the operations in this script define a computation graph that looks like this (where the oval are operations, squares/houses are data):

.. image:: images/intro.svg


Constant operation parameters: ``params``
-----------------------------------------

Sometimes an ``operation`` will have a customizable parameter you want to hold constant across all runs of a computation graph.  Usually, this will be a keyword argument of the underlying function.  The ``params`` argument to the ``operation`` constructor provides a mechanism for setting such parameters.

``params`` should be a dictionary whose keys correspond to keyword parameter names from the function underlying an ``operation`` and whose values are passed as constant arguments to those keyword parameters in all computations utilizing the ``operation``.


Instantiating operations
------------------------

There are several ways to instantiate an ``operation``, each of which might be more suitable for different scenarios.

Decorator specification
^^^^^^^^^^^^^^^^^^^^^^^

If you are defining your computation graph and the functions that comprise it all in the same script, the decorator specification of ``operation`` instances might be particularly useful, as it allows you to assign computation graph structure to functions as they are defined.  Here's an example::

   >>> from graphkit import operation, compose

   >>> @operation(name='foo_op', needs=['a', 'b', 'c'], provides='foo')
   ... def foo(a, b, c):
   ...   return c * (a + b)

   >>> graphop = compose(name='foo_graph')(foo)

Functional specification
^^^^^^^^^^^^^^^^^^^^^^^^

If the functions underlying your computation graph operations are defined elsewhere than the script in which your graph itself is defined (e.g. they are defined in another module, or they are system functions), you can use the functional specification of ``operation`` instances::

   >>> from operator import add, mul
   >>> from graphkit import operation, compose

   >>> add_op = operation(name='add_op', needs=['a', 'b'], provides='sum')(add)
   >>> mul_op = operation(name='mul_op', needs=['c', 'sum'], provides='product')(mul)

   >>> graphop = compose(name='add_mul_graph')(add_op, mul_op)

The functional specification is also useful if you want to create multiple ``operation`` instances from the same function, perhaps with different parameter values, e.g.::

   >>> from graphkit import operation, compose

   >>> def mypow(a, p=2):
   ...    return a ** p

   >>> pow_op1 = operation(name='pow_op1', needs=['a'], provides='a_squared')(mypow)
   >>> pow_op2 = operation(name='pow_op2', needs=['a'], params={'p': 3}, provides='a_cubed')(mypow)

   >>> graphop = compose(name='two_pows_graph')(pow_op1, pow_op2)

A slightly different approach can be used here to accomplish the same effect by creating an operation "factory"::

   from graphkit import operation, compose

   def mypow(a, p=2):
      return a ** p

   pow_op_factory = operation(mypow)

   pow_op1 = pow_op_factory(name='pow_op1', needs=['a'], provides='a_squared')
   pow_op2 = pow_op_factory(name='pow_op2', needs=['a'], params={'p': 3}, provides='a_cubed')

   graphop = compose(name='two_pows_graph')(pow_op1, pow_op2)


Modifiers on ``operation`` inputs and outputs
---------------------------------------------

Certain modifiers are available to apply to input or output values in ``needs`` and ``provides``, for example to designate an optional input.  These modifiers are available in the ``graphkit.modifiers`` module:

.. autoclass:: graphkit.modifiers.optional
.. autoclass:: graphkit.modifiers.sideffect
