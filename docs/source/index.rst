.. graphkit documentation master file, created by
   sphinx-quickstart on Tue Jun 16 19:10:27 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GraphKit
========

GraphKit is a light weight graph processing framework for creating computer vision pipelines and predictive models.

It's simple and fun to use

::

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


It's easy to install

::

    pip install graphkit


What's included in the box?
---------------------------

* Runtime pruning to avoid unnecessary computation
* Parameter serialization
* Multiple inputs and outputs
* DAG visualization


Learn more
----------


.. toctree::
   :maxdepth: 3

   foreword
   getting_started
   reference
   faq

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
