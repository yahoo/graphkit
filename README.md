# GraphKit

[![PyPI version](https://badge.fury.io/py/graphkit.svg)](https://badge.fury.io/py/graphkit) [![Build Status](https://travis-ci.org/yahoo/graphkit.svg?branch=master)](https://travis-ci.org/yahoo/graphkit) [![codecov](https://codecov.io/gh/yahoo/graphkit/branch/master/graph/badge.svg)](https://codecov.io/gh/yahoo/graphkit)

[Full Documentation](https://pythonhosted.org/graphkit/)

> It's a DAG all the way down

![Sample graph](docs/source/images/test_pruning_not_overrides_given_intermediate-asked.png "Sample graph")

## Lightweight computation graphs for Python

GraphKit is a lightweight Python module for creating and running ordered graphs of computations, where the nodes of the graph correspond to computational operations, and the edges correspond to output --> input dependencies between those operations.  Such graphs are useful in computer vision, machine learning, and many other domains.

## Quick start

Here's how to install:

    pip install graphkit

OR with dependencies for plotting support (and you need to install [`Graphviz`](https://graphviz.org)
program separately with your OS tools)::

    pip install graphkit[plot]

Here's a Python script with an example GraphKit computation graph that produces multiple outputs (`a * b`, `a - a * b`, and `abs(a - a * b) ** 3`):

```
from operator import mul, sub
from graphkit import compose, operation

# Computes |a|^p.
def abspow(a, p):
    c = abs(a) ** p
    return c

# Compose the mul, sub, and abspow operations into a computation graph.
graphop = compose(name="graphop")(
    operation(name="mul1", needs=["a", "b"], provides=["ab"])(mul),
    operation(name="sub1", needs=["a", "ab"], provides=["a_minus_ab"])(sub),
    operation(name="abspow1", needs=["a_minus_ab"], provides=["abs_a_minus_ab_cubed"], params={"p": 3})(abspow)
)

# Run the graph and request all of the outputs.
out = graphop({'a': 2, 'b': 5})

# Prints "{'a': 2, 'a_minus_ab': -8, 'b': 5, 'ab': 10, 'abs_a_minus_ab_cubed': 512}".
print(out)

# Run the graph and request a subset of the outputs.
out = graphop({'a': 2, 'b': 5}, outputs=["a_minus_ab"])

# Prints "{'a_minus_ab': -8}".
print(out)
```

As you can see, any function can be used as an operation in GraphKit, even ones imported from system modules!


## Plotting

For debugging the above graph-operation you may plot it using these methods:

```python
   graphop.plot(show=True, solution=out)  # open a matplotlib window with solution values in nodes
   graphop.plot("intro.svg")              # other supported formats: png, jpg, pdf, ...
```

![Intro graph](docs/source/images/intro.png "Intro graph")
![Graphkit Legend](docs/source/images/GraphkitLegend.svg "Graphkit Legend")

> **TIP:** The `pydot.Dot` instances returned by `plot()` are rendered as SVG in *Jupyter/IPython*.

# License

Code licensed under the Apache License, Version 2.0 license. See LICENSE file for terms.
