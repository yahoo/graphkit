## Lightweight graph processing for computer vision

> It's a DAG all the way down

GraphKit is a lightweight graph processing engine for creating computer vision pipelines and predictive models.

Install:

```
pip install graphkit
```    

example.py

```
from vision.graphkit import compose, operation

def mul(a, b):
    c = a*b
    return c

def sub(a, b):
    c = a - b
    return c

def pow(a, power):
    c = a**power
    return c

net = compose(name="net")(
    operation(name="mul1", needs=["a", "b"], provides=["a_times_b"])(mul),
    operation(name="sub1", needs=["a", "a_times_b"], provides=["d"])(sub),
    operation(name="pow1", needs=["b"], params={"power": 2}, provides=["e"])(pow)
)

# run your pipeline and request some inputs
out = net({'a': 2, 'b': 5}, ["e"])
print(out)

# run your pipeline and request all outputs
out = net({'a': 2, 'b': 5})
print(out)

```

# License

Code licensed under the Apache License, Version 2.0 license. See LICENSE file for terms.
