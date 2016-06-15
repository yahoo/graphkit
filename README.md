
## Lightweight graph processing for computer vision

> It's a DAG all the way down

GraphKit is a lightweight graph processing engine for creating computer vision pipelines and predictive models.

Install:

```
pip install graphkit
```    


example.py

```
import numpy as np
from flickr.vision.graphkit import network, Operation

# implement an image transformation
class ImageDiff(Operation):
    def compute(self, inputs):
        a = inputs[0]
        b = inputs[1]
        c = a - b
        return [c]

if __name__ == "__main__":

    # describe what data your operation needs and provides
    diff_op = ImageDiff(
        name="diff-op",
        needs=["arr-a", "arr-b"],
        provides=["diff-img"]
    )

    # compile it into a processing pipeline
    net = network.Network()
    net.add_op(diff_op)
    net.compile()

    # run your pipeline with concrete data
    out = net.compute(
        outputs=["diff-img"],
        named_inputs={
            "arr-a": np.random.rand(5, 5),
            "arr-b": np.random.rand(5, 5)
        }
    )
    print(out)
```



# License

Code licensed under the Apache License, Version 2.0 license. See LICENSE file for terms.
