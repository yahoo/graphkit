# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

import math
import pickle

from pprint import pprint
from operator import add
from numpy.testing import assert_raises

import graphkit.network as network
import graphkit.modifiers as modifiers
from graphkit import operation, compose, Operation

def test_network():

    # Sum operation, late-bind compute function
    sum_op1 = operation(name='sum_op1', needs=['a', 'b'], provides='sum_ab')(add)

    # sum_op1 is callable
    print(sum_op1(1, 2))

    # Multiply operation, decorate in-place
    @operation(name='mul_op1', needs=['sum_ab', 'b'], provides='sum_ab_times_b')
    def mul_op1(a, b):
        return a * b

    # mul_op1 is callable
    print(mul_op1(1, 2))

    # Pow operation
    @operation(name='pow_op1', needs='sum_ab', provides=['sum_ab_p1', 'sum_ab_p2', 'sum_ab_p3'], params={'exponent': 3})
    def pow_op1(a, exponent=2):
        return [math.pow(a, y) for y in range(1, exponent+1)]

    print(pow_op1._compute({'sum_ab':2}, ['sum_ab_p2']))

    # Partial operation that is bound at a later time
    partial_op = operation(name='sum_op2', needs=['sum_ab_p1', 'sum_ab_p2'], provides='p1_plus_p2')

    # Bind the partial operation
    sum_op2 = partial_op(add)

    # Sum operation, early-bind compute function
    sum_op_factory = operation(add)

    sum_op3 = sum_op_factory(name='sum_op3', needs=['a', 'b'], provides='sum_ab2')

    # sum_op3 is callable
    print(sum_op3(5, 6))

    # compose network
    net = compose(name='my network')(sum_op1, mul_op1, pow_op1, sum_op2, sum_op3)

    #
    # Running the network
    #

    # get all outputs
    pprint(net({'a': 1, 'b': 2}))

    # get specific outputs
    pprint(net({'a': 1, 'b': 2}, outputs=["sum_ab_times_b"]))

    # start with inputs already computed
    pprint(net({"sum_ab": 1, "b": 2}, outputs=["sum_ab_times_b"]))

    # visualize network graph
    # net.plot(show=True)


def test_network_simple_merge():

    sum_op1 = operation(name='sum_op1', needs=['a', 'b'], provides='sum1')(add)
    sum_op2 = operation(name='sum_op2', needs=['a', 'b'], provides='sum2')(add)
    sum_op3 = operation(name='sum_op3', needs=['sum1', 'c'], provides='sum3')(add)
    net1 = compose(name='my network 1')(sum_op1, sum_op2, sum_op3)
    pprint(net1({'a': 1, 'b': 2, 'c': 4}))

    sum_op4 = operation(name='sum_op1', needs=['d', 'e'], provides='a')(add)
    sum_op5 = operation(name='sum_op2', needs=['a', 'f'], provides='b')(add)
    net2 = compose(name='my network 2')(sum_op4, sum_op5)
    pprint(net2({'d': 1, 'e': 2, 'f': 4}))

    net3 = compose(name='merged')(net1, net2)
    pprint(net3({'c': 5, 'd': 1, 'e': 2, 'f': 4}))


def test_network_deep_merge():

    sum_op1 = operation(name='sum_op1', needs=['a', 'b'], provides='sum1')(add)
    sum_op2 = operation(name='sum_op2', needs=['a', 'b'], provides='sum2')(add)
    sum_op3 = operation(name='sum_op3', needs=['sum1', 'c'], provides='sum3')(add)
    net1 = compose(name='my network 1')(sum_op1, sum_op2, sum_op3)
    pprint(net1({'a': 1, 'b': 2, 'c': 4}))

    sum_op4 = operation(name='sum_op1', needs=['a', 'b'], provides='sum1')(add)
    sum_op5 = operation(name='sum_op4', needs=['sum1', 'b'], provides='sum2')(add)
    net2 = compose(name='my network 2')(sum_op4, sum_op5)
    pprint(net2({'a': 1, 'b': 2}))

    net3 = compose(name='merged', merge=True)(net1, net2)
    pprint(net3({'a': 1, 'b': 2, 'c': 4}))


def test_input_based_pruning():
    # Tests to make sure we don't need to pass graph inputs if we're provided
    # with data further downstream in the graph as an input.

    sum1 = 2
    sum2 = 5

    # Set up a net such that if sum1 and sum2 are provided directly, we don't
    # need to provide a and b.
    sum_op1 = operation(name='sum_op1', needs=['a', 'b'], provides='sum1')(add)
    sum_op2 = operation(name='sum_op2', needs=['a', 'b'], provides='sum2')(add)
    sum_op3 = operation(name='sum_op3', needs=['sum1', 'sum2'], provides='sum3')(add)
    net = compose(name='test_net')(sum_op1, sum_op2, sum_op3)

    results = net({'sum1': sum1, 'sum2': sum2})

    # Make sure we got expected result without having to pass a or b.
    assert 'sum3' in results
    assert results['sum3'] == add(sum1, sum2)


def test_output_based_pruning():
    # Tests to make sure we don't need to pass graph inputs if they're not
    # needed to compute the requested outputs.

    c = 2
    d = 3

    # Set up a network such that we don't need to provide a or b if we only
    # request sum3 as output.
    sum_op1 = operation(name='sum_op1', needs=['a', 'b'], provides='sum1')(add)
    sum_op2 = operation(name='sum_op2', needs=['c', 'd'], provides='sum2')(add)
    sum_op3 = operation(name='sum_op3', needs=['c', 'sum2'], provides='sum3')(add)
    net = compose(name='test_net')(sum_op1, sum_op2, sum_op3)

    results = net({'c': c, 'd': d}, outputs=['sum3'])

    # Make sure we got expected result without having to pass a or b.
    assert 'sum3' in results
    assert results['sum3'] == add(c, add(c, d))


def test_input_output_based_pruning():
    # Tests to make sure we don't need to pass graph inputs if they're not
    # needed to compute the requested outputs or of we're provided with
    # inputs that are further downstream in the graph.

    c = 2
    sum2 = 5

    # Set up a network such that we don't need to provide a or b d if we only
    # request sum3 as output and if we provide sum2.
    sum_op1 = operation(name='sum_op1', needs=['a', 'b'], provides='sum1')(add)
    sum_op2 = operation(name='sum_op2', needs=['c', 'd'], provides='sum2')(add)
    sum_op3 = operation(name='sum_op3', needs=['c', 'sum2'], provides='sum3')(add)
    net = compose(name='test_net')(sum_op1, sum_op2, sum_op3)

    results = net({'c': c, 'sum2': sum2}, outputs=['sum3'])

    # Make sure we got expected result without having to pass a, b, or d.
    assert 'sum3' in results
    assert results['sum3'] == add(c, sum2)


def test_pruning_raises_for_bad_output():
    # Make sure we get a ValueError during the pruning step if we request an
    # output that doesn't exist.

    # Set up a network that doesn't have the output sum4, which we'll request
    # later.
    sum_op1 = operation(name='sum_op1', needs=['a', 'b'], provides='sum1')(add)
    sum_op2 = operation(name='sum_op2', needs=['c', 'd'], provides='sum2')(add)
    sum_op3 = operation(name='sum_op3', needs=['c', 'sum2'], provides='sum3')(add)
    net = compose(name='test_net')(sum_op1, sum_op2, sum_op3)

    # Request two outputs we can compute and one we can't compute.  Assert
    # that this raises a ValueError.
    assert_raises(ValueError, net, {'a': 1, 'b': 2, 'c': 3, 'd': 4},
        outputs=['sum1', 'sum3', 'sum4'])


def test_optional():
    # Test that optional() needs work as expected.

    # Function to add two values plus an optional third value.
    def addplusplus(a, b, c=0):
        return a + b + c

    sum_op = operation(name='sum_op1', needs=['a', 'b', modifiers.optional('c')], provides='sum')(addplusplus)

    net = compose(name='test_net')(sum_op)

    # Make sure output with optional arg is as expected.
    named_inputs = {'a': 4, 'b': 3, 'c': 2}
    results = net(named_inputs)
    assert 'sum' in results
    assert results['sum'] == sum(named_inputs.values())

    # Make sure output without optional arg is as expected.
    named_inputs = {'a': 4, 'b': 3}
    results = net(named_inputs)
    assert 'sum' in results
    assert results['sum'] == sum(named_inputs.values())


def test_deleted_optional():
    # Test that DeleteInstructions included for optionals do not raise
    # exceptions when the corresponding input is not prodided.

    # Function to add two values plus an optional third value.
    def addplusplus(a, b, c=0):
        return a + b + c

    # Here, a DeleteInstruction will be inserted for the optional need 'c'.
    sum_op1 = operation(name='sum_op1', needs=['a', 'b', modifiers.optional('c')], provides='sum1')(addplusplus)
    sum_op2 = operation(name='sum_op2', needs=['sum1', 'sum1'], provides='sum2')(add)
    net = compose(name='test_net')(sum_op1, sum_op2)

    # DeleteInstructions are used only when a subset of outputs are requested.
    results = net({'a': 4, 'b': 3}, outputs=['sum2'])
    assert 'sum2' in results



def test_parallel_execution():
    import time

    def fn(x):
        time.sleep(1)
        print("fn %s" % (time.time() - t0))
        return 1 + x

    def fn2(a,b):
        time.sleep(1)
        print("fn2 %s" % (time.time() - t0))
        return a+b

    def fn3(z, k=1):
        time.sleep(1)
        print("fn3 %s" % (time.time() - t0))
        return z + k

    pipeline = compose(name="l", merge=True)(

        # the following should execute in parallel under threaded execution mode
        operation(name="a", needs="x", provides="ao")(fn),
        operation(name="b", needs="x", provides="bo")(fn),

        # this should execute after a and b have finished
        operation(name="c", needs=["ao", "bo"], provides="co")(fn2),

        operation(name="d",
                  needs=["ao", modifiers.optional("k")],
                  provides="do")(fn3),

        operation(name="e", needs=["ao", "bo"], provides="eo")(fn2),
        operation(name="f", needs="eo", provides="fo")(fn),
        operation(name="g", needs="fo", provides="go")(fn)


    )

    t0 = time.time()
    pipeline.set_execution_method("parallel")
    result_threaded = pipeline({"x": 10}, ["co", "go", "do"])
    print("threaded result")
    print(result_threaded)

    t0 = time.time()
    pipeline.set_execution_method("sequential")
    result_sequential = pipeline({"x": 10}, ["co", "go", "do"])
    print("sequential result")
    print(result_sequential)

    # make sure results are the same using either method
    assert result_sequential == result_threaded

def test_multi_threading():
    import time
    import random
    from multiprocessing.dummy import Pool

    def op_a(a, b):
        time.sleep(random.random()*.02)
        return a+b

    def op_b(c, b):
        time.sleep(random.random()*.02)
        return c+b

    def op_c(a, b):
        time.sleep(random.random()*.02)
        return a*b

    pipeline = compose(name="pipeline", merge=True)(
        operation(name="op_a", needs=['a', 'b'], provides='c')(op_a),
        operation(name="op_b", needs=['c', 'b'], provides='d')(op_b),
        operation(name="op_c", needs=['a', 'b'], provides='e')(op_c),
    )

    def infer(i):
        # data = open("616039-bradpitt.jpg").read()
        outputs = ["c", "d", "e"]
        results = pipeline({"a": 1, "b":2}, outputs)
        assert tuple(sorted(results.keys())) == tuple(sorted(outputs)), (outputs, results)
        return results

    N = 100
    for i in range(20, 200):
        pool = Pool(i)
        pool.map(infer, range(N))
        pool.close()


####################################
# Backwards compatibility
####################################

# Classes must be defined as members of __main__ for pickleability

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

        a = inputs[0]
        outputs = []
        for y in range(1, self.params['exponent']+1):
            p = math.pow(a, y)
            outputs.append(p)
        return outputs

def test_backwards_compatibility():

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

    # try the pickling part
    pickle.dumps(net)

    #
    # Running the network
    #

    # get all outputs
    pprint(net.compute(outputs=None, named_inputs={'a': 1, 'b': 2}))

    # get specific outputs
    pprint(net.compute(outputs=["sum_ab_times_b"], named_inputs={'a': 1, 'b': 2}))

    # start with inputs already computed
    pprint(net.compute(outputs=["sum_ab_times_b"], named_inputs={"sum_ab": 1, "b": 2}))
