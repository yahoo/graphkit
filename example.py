# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

from flickr.vision.graphkit import Operation
from flickr.vision.graphkit import network


# We first define some basic operations
class Sum(Operation):

    def compute(self, inputs):
        a = inputs[0]
        b = inputs[1]
        return [a+b]


class Mul(Operation):

    def compute(self, inputs):
        a = inputs[0]
        b = inputs[1]
        return [a*b]


# This is an example of an operation that takes a parameter.
# It also illustrates an operation that returns multiple outputs
class Pow(Operation):

    def compute(self, inputs):
        import math

        a = inputs[0]
        outputs = []
        for y in range(1, self.params['exponent']+1):
            p = math.pow(a, y)
            outputs.append(p)
        return outputs


if __name__ == '__main__':

    # We now instantiate multiple operations and define
    # the data flow between each operation

    sum_op1 = Sum(
        name="sum_op1",
        provides=["sum_ab"],
        needs=["a", "b"]
    )
    mul_op1 = Mul(
        name="mul_op1",
        provides=["sum_ab_times_b"],
        needs=["sum_ab", "b"]
    )
    pow_op1 = Pow(
        name="pow_op1",
        needs=["sum_ab"],
        provides=["sum_ab_p1", "sum_ab_p2", "sum_ab_p3"],
        params={"exponent": 3}
    )
    sum_op2 = Sum(
        name="sum_op2",
        provides=["p1_plus_p2"],
        needs=["sum_ab_p1", "sum_ab_p2"],
    )

    net = network.Network()
    net.add_op(sum_op1)
    net.add_op(mul_op1)
    net.add_op(pow_op1)
    net.add_op(sum_op2)
    net.compile()

    #
    # Running the network
    #
    from pprint import pprint

    # get all outputs
    pprint(net.compute(outputs=network.ALL_OUTPUTS, named_inputs={"a": 1, "b": 2}))

    # get specific outputs
    pprint(net.compute(outputs=["sum_ab_times_b"], named_inputs={"a": 1, "b": 2}))

    # start with inputs already computed
    pprint(net.compute(outputs=["sum_ab_times_b"], named_inputs={"sum_ab": 1, "b": 2}))

    # visualize network graph
    net.plot(show=True, filename="out.png")



    # Serialization
    # import pickle
    #
    # # seriliaze entire network
    # pickle.dump(net, open("net.pkl", "wb"), pickle.HIGHEST_PROTOCOL)
    #
    # # serialize individual operations
    # pickle.dump(pow_op1, open("pow_op1.pkl", "wb"), pickle.HIGHEST_PROTOCOL)
