# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

import math
import pickle
import sys
from operator import add, floordiv, mul, sub
from pprint import pprint

import pytest

import graphkit.network as network
from graphkit import Operation, compose, operation, optional, sideffect
from graphkit.network import _EvictInstruction


def scream(*args, **kwargs):
    raise AssertionError(
        "Must not have run!\n    args: %s\n  kwargs: %s", (args, kwargs)
    )


def identity(x):
    return x


def filtdict(d, *keys):
    """
    Keep dict items with the given keys

    >>> filtdict({"a": 1, "b": 2}, "b")
    {'b': 2}
    """
    return type(d)(i for i in d.items() if i[0] in keys)


def test_network_smoke():

    # Sum operation, late-bind compute function
    sum_op1 = operation(name="sum_op1", needs=["a", "b"], provides="sum_ab")(add)

    # sum_op1 is callable
    assert sum_op1(1, 2) == 3

    # Multiply operation, decorate in-place
    @operation(name="mul_op1", needs=["sum_ab", "b"], provides="sum_ab_times_b")
    def mul_op1(a, b):
        return a * b

    # mul_op1 is callable
    assert mul_op1(1, 2) == 2

    # Pow operation
    @operation(
        name="pow_op1",
        needs="sum_ab",
        provides=["sum_ab_p1", "sum_ab_p2", "sum_ab_p3"],
        params={"exponent": 3},
    )
    def pow_op1(a, exponent=2):
        return [math.pow(a, y) for y in range(1, exponent + 1)]

    assert pow_op1._compute({"sum_ab": 2}, ["sum_ab_p2"]) == {"sum_ab_p2": 4.0}

    # Partial operation that is bound at a later time
    partial_op = operation(
        name="sum_op2", needs=["sum_ab_p1", "sum_ab_p2"], provides="p1_plus_p2"
    )

    # Bind the partial operation
    sum_op2 = partial_op(add)

    # Sum operation, early-bind compute function
    sum_op_factory = operation(add)

    sum_op3 = sum_op_factory(name="sum_op3", needs=["a", "b"], provides="sum_ab2")

    # sum_op3 is callable
    assert sum_op3(5, 6) == 11

    # compose network
    net = compose(name="my network")(sum_op1, mul_op1, pow_op1, sum_op2, sum_op3)

    #
    # Running the network
    #

    # get all outputs
    exp = {
        "a": 1,
        "b": 2,
        "p1_plus_p2": 12.0,
        "sum_ab": 3,
        "sum_ab2": 3,
        "sum_ab_p1": 3.0,
        "sum_ab_p2": 9.0,
        "sum_ab_p3": 27.0,
        "sum_ab_times_b": 6,
    }
    assert net({"a": 1, "b": 2}) == exp

    # get specific outputs
    exp = {"sum_ab_times_b": 6}
    assert net({"a": 1, "b": 2}, outputs=["sum_ab_times_b"]) == exp

    # start with inputs already computed
    exp = {"sum_ab_times_b": 2}
    assert net({"sum_ab": 1, "b": 2}, outputs=["sum_ab_times_b"]) == exp

    with pytest.raises(ValueError, match="Unknown output node"):
        net({"sum_ab": 1, "b": 2}, outputs="bad_node")
    with pytest.raises(ValueError, match="Unknown output node"):
        net({"sum_ab": 1, "b": 2}, outputs=["b", "bad_node"])


def test_network_simple_merge():

    sum_op1 = operation(name="sum_op1", needs=["a", "b"], provides="sum1")(add)
    sum_op2 = operation(name="sum_op2", needs=["a", "b"], provides="sum2")(add)
    sum_op3 = operation(name="sum_op3", needs=["sum1", "c"], provides="sum3")(add)
    net1 = compose(name="my network 1")(sum_op1, sum_op2, sum_op3)

    exp = {"a": 1, "b": 2, "c": 4, "sum1": 3, "sum2": 3, "sum3": 7}
    sol = net1({"a": 1, "b": 2, "c": 4})
    assert sol == exp

    sum_op4 = operation(name="sum_op1", needs=["d", "e"], provides="a")(add)
    sum_op5 = operation(name="sum_op2", needs=["a", "f"], provides="b")(add)

    net2 = compose(name="my network 2")(sum_op4, sum_op5)
    exp = {"a": 3, "b": 7, "d": 1, "e": 2, "f": 4}
    sol = net2({"d": 1, "e": 2, "f": 4})
    assert sol == exp

    net3 = compose(name="merged")(net1, net2)
    exp = {
        "a": 3,
        "b": 7,
        "c": 5,
        "d": 1,
        "e": 2,
        "f": 4,
        "sum1": 10,
        "sum2": 10,
        "sum3": 15,
    }
    sol = net3({"c": 5, "d": 1, "e": 2, "f": 4})
    assert sol == exp


def test_network_deep_merge():

    sum_op1 = operation(name="sum_op1", needs=["a", "b"], provides="sum1")(add)
    sum_op2 = operation(name="sum_op2", needs=["a", "b"], provides="sum2")(add)
    sum_op3 = operation(name="sum_op3", needs=["sum1", "c"], provides="sum3")(add)
    net1 = compose(name="my network 1")(sum_op1, sum_op2, sum_op3)

    exp = {"a": 1, "b": 2, "c": 4, "sum1": 3, "sum2": 3, "sum3": 7}
    assert net1({"a": 1, "b": 2, "c": 4}) == exp

    sum_op4 = operation(name="sum_op1", needs=["a", "b"], provides="sum1")(add)
    sum_op5 = operation(name="sum_op4", needs=["sum1", "b"], provides="sum2")(add)
    net2 = compose(name="my network 2")(sum_op4, sum_op5)
    exp = {"a": 1, "b": 2, "sum1": 3, "sum2": 5}
    assert net2({"a": 1, "b": 2}) == exp

    net3 = compose(name="merged", merge=True)(net1, net2)
    exp = {"a": 1, "b": 2, "c": 4, "sum1": 3, "sum2": 3, "sum3": 7}
    assert net3({"a": 1, "b": 2, "c": 4}) == exp


def test_network_merge_in_doctests():
    def abspow(a, p):
        c = abs(a) ** p
        return c

    graphop = compose(name="graphop")(
        operation(name="mul1", needs=["a", "b"], provides=["ab"])(mul),
        operation(name="sub1", needs=["a", "ab"], provides=["a_minus_ab"])(sub),
        operation(
            name="abspow1",
            needs=["a_minus_ab"],
            provides=["abs_a_minus_ab_cubed"],
            params={"p": 3},
        )(abspow),
    )

    another_graph = compose(name="another_graph")(
        operation(name="mul1", needs=["a", "b"], provides=["ab"])(mul),
        operation(name="mul2", needs=["c", "ab"], provides=["cab"])(mul),
    )
    merged_graph = compose(name="merged_graph", merge=True)(graphop, another_graph)
    assert merged_graph.needs
    assert merged_graph.provides


def test_input_based_pruning():
    # Tests to make sure we don't need to pass graph inputs if we're provided
    # with data further downstream in the graph as an input.

    sum1 = 2
    sum2 = 5

    # Set up a net such that if sum1 and sum2 are provided directly, we don't
    # need to provide a and b.
    sum_op1 = operation(name="sum_op1", needs=["a", "b"], provides="sum1")(add)
    sum_op2 = operation(name="sum_op2", needs=["a", "b"], provides="sum2")(add)
    sum_op3 = operation(name="sum_op3", needs=["sum1", "sum2"], provides="sum3")(add)
    net = compose(name="test_net")(sum_op1, sum_op2, sum_op3)

    results = net({"sum1": sum1, "sum2": sum2})

    # Make sure we got expected result without having to pass a or b.
    assert "sum3" in results
    assert results["sum3"] == add(sum1, sum2)


def test_output_based_pruning():
    # Tests to make sure we don't need to pass graph inputs if they're not
    # needed to compute the requested outputs.

    c = 2
    d = 3

    # Set up a network such that we don't need to provide a or b if we only
    # request sum3 as output.
    sum_op1 = operation(name="sum_op1", needs=["a", "b"], provides="sum1")(add)
    sum_op2 = operation(name="sum_op2", needs=["c", "d"], provides="sum2")(add)
    sum_op3 = operation(name="sum_op3", needs=["c", "sum2"], provides="sum3")(add)
    net = compose(name="test_net")(sum_op1, sum_op2, sum_op3)

    results = net({"a": 0, "b": 0, "c": c, "d": d}, outputs=["sum3"])

    # Make sure we got expected result without having to pass a or b.
    assert "sum3" in results
    assert results["sum3"] == add(c, add(c, d))


def test_input_output_based_pruning():
    # Tests to make sure we don't need to pass graph inputs if they're not
    # needed to compute the requested outputs or of we're provided with
    # inputs that are further downstream in the graph.

    c = 2
    sum2 = 5

    # Set up a network such that we don't need to provide a or b d if we only
    # request sum3 as output and if we provide sum2.
    sum_op1 = operation(name="sum_op1", needs=["a", "b"], provides="sum1")(add)
    sum_op2 = operation(name="sum_op2", needs=["c", "d"], provides="sum2")(add)
    sum_op3 = operation(name="sum_op3", needs=["c", "sum2"], provides="sum3")(add)
    net = compose(name="test_net")(sum_op1, sum_op2, sum_op3)

    results = net({"c": c, "sum2": sum2}, outputs=["sum3"])

    # Make sure we got expected result without having to pass a, b, or d.
    assert "sum3" in results
    assert results["sum3"] == add(c, sum2)


def test_pruning_raises_for_bad_output():
    # Make sure we get a ValueError during the pruning step if we request an
    # output that doesn't exist.

    # Set up a network that doesn't have the output sum4, which we'll request
    # later.
    sum_op1 = operation(name="sum_op1", needs=["a", "b"], provides="sum1")(add)
    sum_op2 = operation(name="sum_op2", needs=["c", "d"], provides="sum2")(add)
    sum_op3 = operation(name="sum_op3", needs=["c", "sum2"], provides="sum3")(add)
    net = compose(name="test_net")(sum_op1, sum_op2, sum_op3)

    # Request two outputs we can compute and one we can't compute.  Assert
    # that this raises a ValueError.
    with pytest.raises(ValueError) as exinfo:
        net({"a": 1, "b": 2, "c": 3, "d": 4}, outputs=["sum1", "sum3", "sum4"])
    assert exinfo.match("sum4")


def test_pruning_not_overrides_given_intermediate():
    # Test #25: v1.2.4 overwrites intermediate data when no output asked
    pipeline = compose(name="pipeline")(
        operation(name="not run", needs=["a"], provides=["overriden"])(scream),
        operation(name="op", needs=["overriden", "c"], provides=["asked"])(add),
    )

    inputs = {"a": 5, "overriden": 1, "c": 2}
    exp = {"a": 5, "overriden": 1, "c": 2, "asked": 3}
    # v1.2.4.ok
    assert pipeline(inputs, ["asked"]) == filtdict(exp, "asked")
    # FAILs
    # - on v1.2.4 with (overriden, asked): = (5, 7) instead of (1, 3)
    # - on #18(unsatisfied) + #23(ordered-sets) with (overriden, asked) = (5, 7) instead of (1, 3)
    # FIXED on #26
    assert pipeline(inputs) == exp

    ## Test OVERWITES
    #
    overwrites = {}
    pipeline.set_overwrites_collector(overwrites)
    assert pipeline(inputs, ["asked"]) == filtdict(exp, "asked")
    assert overwrites == {}  # unjust must have been pruned

    overwrites = {}
    pipeline.set_overwrites_collector(overwrites)
    assert pipeline(inputs) == exp
    assert overwrites == {}  # unjust must have been pruned

    ## Test Parallel
    #
    pipeline.set_execution_method("parallel")
    overwrites = {}
    pipeline.set_overwrites_collector(overwrites)
    # assert pipeline(inputs, ["asked"]) == filtdict(exp, "asked")
    assert overwrites == {}  # unjust must have been pruned

    overwrites = {}
    pipeline.set_overwrites_collector(overwrites)
    assert pipeline(inputs) == exp
    assert overwrites == {}  # unjust must have been pruned


def test_pruning_multiouts_not_override_intermediates1():
    # Test #25: v.1.2.4 overwrites intermediate data when a previous operation
    # must run for its other outputs (outputs asked or not)
    pipeline = compose(name="graph")(
        operation(name="must run", needs=["a"], provides=["overriden", "calced"])(
            lambda x: (x, 2 * x)
        ),
        operation(name="add", needs=["overriden", "calced"], provides=["asked"])(add),
    )

    inputs = {"a": 5, "overriden": 1, "c": 2}
    exp = {"a": 5, "overriden": 1, "calced": 10, "asked": 11}
    # FAILs
    # - on v1.2.4 with (overriden, asked) = (5, 15) instead of (1, 11)
    # - on #18(unsatisfied) + #23(ordered-sets) like v1.2.4.
    # FIXED on #26
    assert pipeline({"a": 5, "overriden": 1}) == exp
    # FAILs
    # - on v1.2.4 with KeyError: 'e',
    # - on #18(unsatisfied) + #23(ordered-sets) with empty result.
    # FIXED on #26
    assert pipeline(inputs, ["asked"]) == filtdict(exp, "asked")
    # Plan must contain "overriden" step twice, for pin & evict.
    # Plot it to see, or check https://github.com/huyng/graphkit/pull/1#discussion_r334226396.
    datasteps = [s for s in pipeline.net.last_plan.steps if s == "overriden"]
    assert len(datasteps) == 2
    assert isinstance(datasteps[0], network._PinInstruction)
    assert isinstance(datasteps[1], network._EvictInstruction)

    ## Test OVERWITES
    #
    overwrites = {}
    pipeline.set_overwrites_collector(overwrites)
    assert pipeline({"a": 5, "overriden": 1}) == exp
    assert overwrites == {"overriden": 5}

    overwrites = {}
    pipeline.set_overwrites_collector(overwrites)
    assert pipeline(inputs, ["asked"]) == filtdict(exp, "asked")
    assert overwrites == {"overriden": 5}

    ## Test parallel
    #
    pipeline.set_execution_method("parallel")
    assert pipeline({"a": 5, "overriden": 1}) == exp
    assert pipeline(inputs, ["asked"]) == filtdict(exp, "asked")


@pytest.mark.xfail(
    sys.version_info < (3, 6),
    reason="PY3.5- have unstable dicts."
    "E.g. https://travis-ci.org/ankostis/graphkit/jobs/595841023",
)
def test_pruning_multiouts_not_override_intermediates2():
    # Test #25: v.1.2.4 overrides intermediate data when a previous operation
    # must run for its other outputs (outputs asked or not)
    # SPURIOUS FAILS in < PY3.6 due to unordered dicts,
    # eg https://travis-ci.org/ankostis/graphkit/jobs/594813119
    pipeline = compose(name="pipeline")(
        operation(name="must run", needs=["a"], provides=["overriden", "e"])(
            lambda x: (x, 2 * x)
        ),
        operation(name="op1", needs=["overriden", "c"], provides=["d"])(add),
        operation(name="op2", needs=["d", "e"], provides=["asked"])(mul),
    )

    inputs = {"a": 5, "overriden": 1, "c": 2}
    exp = {"a": 5, "overriden": 1, "c": 2, "d": 3, "e": 10, "asked": 30}
    # FAILs
    # - on v1.2.4 with (overriden, asked) = (5, 70) instead of (1, 13)
    # - on #18(unsatisfied) + #23(ordered-sets) like v1.2.4.
    # FIXED on #26
    assert pipeline(inputs) == exp
    # FAILs
    # - on v1.2.4 with KeyError: 'e',
    # - on #18(unsatisfied) + #23(ordered-sets) with empty result.
    assert pipeline(inputs, ["asked"]) == filtdict(exp, "asked")
    # FIXED on #26

    ## Test OVERWITES
    #
    overwrites = {}
    pipeline.set_overwrites_collector(overwrites)
    assert pipeline(inputs) == exp
    assert overwrites == {"overriden": 5}

    overwrites = {}
    pipeline.set_overwrites_collector(overwrites)
    assert pipeline(inputs, ["asked"]) == filtdict(exp, "asked")
    assert overwrites == {"overriden": 5}

    ## Test parallel
    #
    pipeline.set_execution_method("parallel")
    assert pipeline(inputs) == exp
    assert pipeline(inputs, ["asked"]) == filtdict(exp, "asked")


def test_pruning_with_given_intermediate_and_asked_out():
    # Test #24: v1.2.4 does not prune before given intermediate data when
    # outputs not asked, but does so when output asked.
    pipeline = compose(name="pipeline")(
        operation(name="unjustly pruned", needs=["given-1"], provides=["a"])(identity),
        operation(name="shortcuted", needs=["a", "b"], provides=["given-2"])(add),
        operation(name="good_op", needs=["a", "given-2"], provides=["asked"])(add),
    )

    exp = {"given-1": 5, "b": 2, "given-2": 2, "a": 5, "asked": 7}
    # v1.2.4 is ok
    assert pipeline({"given-1": 5, "b": 2, "given-2": 2}) == exp
    # FAILS
    # - on v1.2.4 with KeyError: 'a',
    # - on #18 (unsatisfied) with no result.
    # FIXED on #18+#26 (new dag solver).
    assert pipeline({"given-1": 5, "b": 2, "given-2": 2}, ["asked"]) == filtdict(
        exp, "asked"
    )

    ## Test OVERWITES
    #
    overwrites = {}
    pipeline.set_overwrites_collector(overwrites)
    assert pipeline({"given-1": 5, "b": 2, "given-2": 2}) == exp
    assert overwrites == {}

    overwrites = {}
    pipeline.set_overwrites_collector(overwrites)
    assert pipeline({"given-1": 5, "b": 2, "given-2": 2}, ["asked"]) == filtdict(
        exp, "asked"
    )
    assert overwrites == {}

    ## Test parallel
    #  FAIL! in #26!
    #
    pipeline.set_execution_method("parallel")
    assert pipeline({"given-1": 5, "b": 2, "given-2": 2}) == exp
    assert pipeline({"given-1": 5, "b": 2, "given-2": 2}, ["asked"]) == filtdict(
        exp, "asked"
    )


def test_unsatisfied_operations():
    # Test that operations with partial inputs are culled and not failing.
    pipeline = compose(name="pipeline")(
        operation(name="add", needs=["a", "b1"], provides=["a+b1"])(add),
        operation(name="sub", needs=["a", "b2"], provides=["a-b2"])(sub),
    )

    exp = {"a": 10, "b1": 2, "a+b1": 12}
    assert pipeline({"a": 10, "b1": 2}) == exp
    assert pipeline({"a": 10, "b1": 2}, outputs=["a+b1"]) == filtdict(exp, "a+b1")

    exp = {"a": 10, "b2": 2, "a-b2": 8}
    assert pipeline({"a": 10, "b2": 2}) == exp
    assert pipeline({"a": 10, "b2": 2}, outputs=["a-b2"]) == filtdict(exp, "a-b2")

    ## Test parallel
    #
    pipeline.set_execution_method("parallel")
    exp = {"a": 10, "b1": 2, "a+b1": 12}
    assert pipeline({"a": 10, "b1": 2}) == exp
    assert pipeline({"a": 10, "b1": 2}, outputs=["a+b1"]) == filtdict(exp, "a+b1")

    exp = {"a": 10, "b2": 2, "a-b2": 8}
    assert pipeline({"a": 10, "b2": 2}) == exp
    assert pipeline({"a": 10, "b2": 2}, outputs=["a-b2"]) == filtdict(exp, "a-b2")


def test_unsatisfied_operations_same_out():
    # Test unsatisfied pairs of operations providing the same output.
    pipeline = compose(name="pipeline")(
        operation(name="mul", needs=["a", "b1"], provides=["ab"])(mul),
        operation(name="div", needs=["a", "b2"], provides=["ab"])(floordiv),
        operation(name="add", needs=["ab", "c"], provides=["ab_plus_c"])(add),
    )

    exp = {"a": 10, "b1": 2, "c": 1, "ab": 20, "ab_plus_c": 21}
    assert pipeline({"a": 10, "b1": 2, "c": 1}) == exp
    assert pipeline({"a": 10, "b1": 2, "c": 1}, outputs=["ab_plus_c"]) == filtdict(
        exp, "ab_plus_c"
    )

    exp = {"a": 10, "b2": 2, "c": 1, "ab": 5, "ab_plus_c": 6}
    assert pipeline({"a": 10, "b2": 2, "c": 1}) == exp
    assert pipeline({"a": 10, "b2": 2, "c": 1}, outputs=["ab_plus_c"]) == filtdict(
        exp, "ab_plus_c"
    )

    ## Test parallel
    #
    #  FAIL! in #26
    pipeline.set_execution_method("parallel")
    exp = {"a": 10, "b1": 2, "c": 1, "ab": 20, "ab_plus_c": 21}
    assert pipeline({"a": 10, "b1": 2, "c": 1}) == exp
    assert pipeline({"a": 10, "b1": 2, "c": 1}, outputs=["ab_plus_c"]) == filtdict(
        exp, "ab_plus_c"
    )
    #
    #  FAIL! in #26
    exp = {"a": 10, "b2": 2, "c": 1, "ab": 5, "ab_plus_c": 6}
    assert pipeline({"a": 10, "b2": 2, "c": 1}) == exp
    assert pipeline({"a": 10, "b2": 2, "c": 1}, outputs=["ab_plus_c"]) == filtdict(
        exp, "ab_plus_c"
    )


def test_optional():
    # Test that optional() needs work as expected.

    # Function to add two values plus an optional third value.
    def addplusplus(a, b, c=0):
        return a + b + c

    sum_op = operation(name="sum_op1", needs=["a", "b", optional("c")], provides="sum")(
        addplusplus
    )

    net = compose(name="test_net")(sum_op)

    # Make sure output with optional arg is as expected.
    named_inputs = {"a": 4, "b": 3, "c": 2}
    results = net(named_inputs)
    assert "sum" in results
    assert results["sum"] == sum(named_inputs.values())

    # Make sure output without optional arg is as expected.
    named_inputs = {"a": 4, "b": 3}
    results = net(named_inputs)
    assert "sum" in results
    assert results["sum"] == sum(named_inputs.values())


def test_sideffects():
    # Function without return value.
    def extend(box):
        box.extend([1, 2])

    def increment(box):
        for i in range(len(box)):
            box[i] += 1

    # Designate `a`, `b` as sideffect inp/out arguments.
    graph = compose("mygraph")(
        operation(
            name="extend",
            needs=["box", sideffect("a")],
            provides=[sideffect("b")],
        )(extend),
        operation(
            name="increment",
            needs=["box", sideffect("b")],
            provides=sideffect("c"),
        )(increment),
    )

    assert graph({"box": [0], "a": True})["box"] == [1, 2, 3]

    # Reverse order of functions.
    graph = compose("mygraph")(
        operation(
            name="increment",
            needs=["box", sideffect("a")],
            provides=sideffect("b"),
        )(increment),
        operation(
            name="extend",
            needs=["box", sideffect("b")],
            provides=[sideffect("c")],
        )(extend),
    )

    assert graph({"box": [0], "a": None})["box"] == [1, 1, 2]


@pytest.mark.xfail(
    sys.version_info < (3, 6),
    reason="PY3.5- have unstable dicts."
    "E.g. https://travis-ci.org/ankostis/graphkit/jobs/595793872",
)
def test_optional_per_function_with_same_output():
    # Test that the same need can be both optional and not on different operations.
    #
    ## ATTENTION, the selected function is NOT the one with more inputs
    # but the 1st satisfiable function added in the network.

    add_op = operation(name="add", needs=["a", "b"], provides="a+-b")(add)
    sub_op_optional = operation(
        name="sub_opt", needs=["a", optional("b")], provides="a+-b"
    )(lambda a, b=10: a - b)

    # Normal order
    #
    pipeline = compose(name="partial_optionals")(add_op, sub_op_optional)
    #
    named_inputs = {"a": 1, "b": 2}
    assert pipeline(named_inputs) == {"a": 1, "a+-b": 3, "b": 2}
    assert pipeline(named_inputs, ["a+-b"]) == {"a+-b": 3}
    #
    named_inputs = {"a": 1}
    assert pipeline(named_inputs) == {"a": 1, "a+-b": -9}
    assert pipeline(named_inputs, ["a+-b"]) == {"a+-b": -9}

    # Inverse op order
    #
    pipeline = compose(name="partial_optionals")(sub_op_optional, add_op)
    #
    named_inputs = {"a": 1, "b": 2}
    assert pipeline(named_inputs) == {"a": 1, "a+-b": -1, "b": 2}
    assert pipeline(named_inputs, ["a+-b"]) == {"a+-b": -1}
    #
    named_inputs = {"a": 1}
    assert pipeline(named_inputs) == {"a": 1, "a+-b": -9}
    assert pipeline(named_inputs, ["a+-b"]) == {"a+-b": -9}

    # PARALLEL + Normal order
    #
    pipeline = compose(name="partial_optionals")(add_op, sub_op_optional)
    pipeline.set_execution_method("parallel")
    #
    named_inputs = {"a": 1, "b": 2}
    assert pipeline(named_inputs) == {"a": 1, "a+-b": 3, "b": 2}
    assert pipeline(named_inputs, ["a+-b"]) == {"a+-b": 3}
    #
    named_inputs = {"a": 1}
    assert pipeline(named_inputs) == {"a": 1, "a+-b": -9}
    assert pipeline(named_inputs, ["a+-b"]) == {"a+-b": -9}

    # PARALLEL + Inverse op order
    #
    pipeline = compose(name="partial_optionals")(sub_op_optional, add_op)
    pipeline.set_execution_method("parallel")
    #
    named_inputs = {"a": 1, "b": 2}
    assert pipeline(named_inputs) == {"a": 1, "a+-b": -1, "b": 2}
    assert pipeline(named_inputs, ["a+-b"]) == {"a+-b": -1}
    #
    named_inputs = {"a": 1}
    assert pipeline(named_inputs) == {"a": 1, "a+-b": -9}
    assert pipeline(named_inputs, ["a+-b"]) == {"a+-b": -9}


def test_evicted_optional():
    # Test that _EvictInstructions included for optionals do not raise
    # exceptions when the corresponding input is not prodided.

    # Function to add two values plus an optional third value.
    def addplusplus(a, b, c=0):
        return a + b + c

    # Here, a _EvictInstruction will be inserted for the optional need 'c'.
    sum_op1 = operation(
        name="sum_op1", needs=["a", "b", optional("c")], provides="sum1"
    )(addplusplus)
    sum_op2 = operation(name="sum_op2", needs=["sum1", "sum1"], provides="sum2")(add)
    net = compose(name="test_net")(sum_op1, sum_op2)

    # _EvictInstructions are used only when a subset of outputs are requested.
    results = net({"a": 4, "b": 3}, outputs=["sum2"])
    assert "sum2" in results


def test_evict_instructions_vary_with_inputs():
    # Check #21: _EvictInstructions positions vary when inputs change.
    def count_evictions(steps):
        return sum(isinstance(n, _EvictInstruction) for n in steps)

    pipeline = compose(name="pipeline")(
        operation(name="a free without b", needs=["a"], provides=["aa"])(identity),
        operation(name="satisfiable", needs=["a", "b"], provides=["ab"])(add),
        operation(name="optional ab", needs=["aa", optional("ab")], provides=["asked"])(
            lambda a, ab=10: a + ab
        ),
    )

    inp = {"a": 2, "b": 3}
    exp = inp.copy()
    exp.update({"aa": 2, "ab": 5, "asked": 7})
    res = pipeline(inp)
    assert res == exp  # ok
    steps11 = pipeline.compile(inp).steps
    res = pipeline(inp, outputs=["asked"])
    assert res == filtdict(exp, "asked")  # ok
    steps12 = pipeline.compile(inp, ["asked"]).steps

    inp = {"a": 2}
    exp = inp.copy()
    exp.update({"aa": 2, "asked": 12})
    res = pipeline(inp)
    assert res == exp  # ok
    steps21 = pipeline.compile(inp).steps
    res = pipeline(inp, outputs=["asked"])
    assert res == filtdict(exp, "asked")  # ok
    steps22 = pipeline.compile(inp, ["asked"]).steps

    # When no outs, no evict-instructions.
    assert steps11 != steps12
    assert count_evictions(steps11) == 0
    assert steps21 != steps22
    assert count_evictions(steps21) == 0

    # Check steps vary with inputs
    #
    # FAILs in v1.2.4 + #18, PASS in #26
    assert steps11 != steps21

    # Check evicts vary with inputs
    #
    # FAILs in v1.2.4 + #18, PASS in #26
    assert count_evictions(steps12) != count_evictions(steps22)


@pytest.mark.slow
def test_parallel_execution():
    import time

    delay = 0.5

    def fn(x):
        time.sleep(delay)
        print("fn %s" % (time.time() - t0))
        return 1 + x

    def fn2(a, b):
        time.sleep(delay)
        print("fn2 %s" % (time.time() - t0))
        return a + b

    def fn3(z, k=1):
        time.sleep(delay)
        print("fn3 %s" % (time.time() - t0))
        return z + k

    pipeline = compose(name="l", merge=True)(
        # the following should execute in parallel under threaded execution mode
        operation(name="a", needs="x", provides="ao")(fn),
        operation(name="b", needs="x", provides="bo")(fn),
        # this should execute after a and b have finished
        operation(name="c", needs=["ao", "bo"], provides="co")(fn2),
        operation(name="d", needs=["ao", optional("k")], provides="do")(fn3),
        operation(name="e", needs=["ao", "bo"], provides="eo")(fn2),
        operation(name="f", needs="eo", provides="fo")(fn),
        operation(name="g", needs="fo", provides="go")(fn),
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


@pytest.mark.slow
def test_multi_threading():
    import time
    import random
    from multiprocessing.dummy import Pool

    def op_a(a, b):
        time.sleep(random.random() * 0.02)
        return a + b

    def op_b(c, b):
        time.sleep(random.random() * 0.02)
        return c + b

    def op_c(a, b):
        time.sleep(random.random() * 0.02)
        return a * b

    pipeline = compose(name="pipeline", merge=True)(
        operation(name="op_a", needs=["a", "b"], provides="c")(op_a),
        operation(name="op_b", needs=["c", "b"], provides="d")(op_b),
        operation(name="op_c", needs=["a", "b"], provides="e")(op_c),
    )

    def infer(i):
        # data = open("616039-bradpitt.jpg").read()
        outputs = ["c", "d", "e"]
        results = pipeline({"a": 1, "b": 2}, outputs)
        assert tuple(sorted(results.keys())) == tuple(sorted(outputs)), (
            outputs,
            results,
        )
        return results

    N = 33
    for i in range(13, 61):
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
        return [a + b]


class Mul(Operation):
    def compute(self, inputs):
        a = inputs[0]
        b = inputs[1]
        return [a * b]


# This is an example of an operation that takes a parameter.
# It also illustrates an operation that returns multiple outputs
class Pow(Operation):
    def compute(self, inputs):

        a = inputs[0]
        outputs = []
        for y in range(1, self.params["exponent"] + 1):
            p = math.pow(a, y)
            outputs.append(p)
        return outputs


def test_backwards_compatibility():

    sum_op1 = Sum(name="sum_op1", provides=["sum_ab"], needs=["a", "b"])
    mul_op1 = Mul(name="mul_op1", provides=["sum_ab_times_b"], needs=["sum_ab", "b"])
    pow_op1 = Pow(
        name="pow_op1",
        needs=["sum_ab"],
        provides=["sum_ab_p1", "sum_ab_p2", "sum_ab_p3"],
        params={"exponent": 3},
    )
    sum_op2 = Sum(
        name="sum_op2", provides=["p1_plus_p2"], needs=["sum_ab_p1", "sum_ab_p2"]
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
    exp = {
        "a": 1,
        "b": 2,
        "p1_plus_p2": 12.0,
        "sum_ab": 3,
        "sum_ab_p1": 3.0,
        "sum_ab_p2": 9.0,
        "sum_ab_p3": 27.0,
        "sum_ab_times_b": 6,
    }
    assert net.compute(outputs=None, named_inputs={"a": 1, "b": 2}) == exp

    # get specific outputs
    exp = {"sum_ab_times_b": 6}
    assert net.compute(outputs=["sum_ab_times_b"], named_inputs={"a": 1, "b": 2}) == exp

    # start with inputs already computed
    exp = {"sum_ab_times_b": 2}
    assert (
        net.compute(outputs=["sum_ab_times_b"], named_inputs={"sum_ab": 1, "b": 2})
        == exp
    )
