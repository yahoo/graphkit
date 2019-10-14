# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import functools as fnt
import itertools as itt
import logging

import pytest

from graphkit import base, network, operation


def test_jetsam_without_failure(caplog):
    caplog.set_level(logging.INFO)
    with pytest.raises(AssertionError, match="No `salvage_mappings`"):
        with base.jetsam({}):
            pytest.xfail("Jetsam did not detect bad inputs!")

    assert "No-op jetsam!  Call" not in caplog.text
    assert "Supressed error" not in caplog.text


@pytest.mark.parametrize("locs", [None, (), [], [0], "bad"])
def test_jetsam_bad_locals(locs, caplog):
    caplog.set_level(logging.INFO)
    with pytest.raises(AssertionError, match="Bad `locs`") as excinfo:
        with base.jetsam(locs, a="a"):
            raise Exception()

    assert not hasattr(excinfo.value, "graphkit_jetsam")
    assert "Supressed error while annotating exception" not in caplog.text


@pytest.mark.parametrize("keys", [{"k": None}, {"k": ()}, {"k": []}, {"k": [0]}])
def test_jetsam_bad_keys(keys, caplog):
    caplog.set_level(logging.INFO)
    with pytest.raises(AssertionError, match="Bad `salvage_mappings`") as excinfo:
        with base.jetsam({}, **keys):
            raise Exception("ABC")

    assert not hasattr(excinfo.value, "graphkit_jetsam")
    assert "Supressed error while annotating exception" not in caplog.text


@pytest.mark.parametrize("locs", [None, (), [], [0], "bad"])
def test_jetsam_bad_locals_given(locs, caplog):
    caplog.set_level(logging.INFO)
    with pytest.raises(AssertionError, match="Bad `locs`") as excinfo:
        with base.jetsam(locs, a="a"):
            raise Exception("ABC")

    assert not hasattr(excinfo.value, "graphkit_jetsam")
    assert "Supressed error while annotating exception" not in caplog.text


@pytest.mark.parametrize("annotation", [None, (), [], [0], "bad"])
def test_jetsam_bad_existing_annotation(annotation, caplog):
    caplog.set_level(logging.INFO)
    with pytest.raises(Exception, match="ABC") as excinfo:
        with base.jetsam({}, a="a"):
            ex = Exception("ABC")
            ex.graphkit_jetsam = annotation
            raise ex

    assert excinfo.value.graphkit_jetsam == {"a": None}
    assert "Supressed error while annotating exception" not in caplog.text


def test_jetsam_dummy_locals(caplog):
    with pytest.raises(Exception, match="ABC") as excinfo:
        with base.jetsam({"a": 1}, a="a", bad="bad"):

            raise Exception("ABC")

    assert isinstance(excinfo.value.graphkit_jetsam, dict)
    assert excinfo.value.graphkit_jetsam == {"a": 1, "bad": None}
    assert "Supressed error" not in caplog.text


def _scream(*args, **kwargs):
    raise Exception("ABC")


def _jetsamed_fn(*args, **kwargs):
    b = 1
    with base.jetsam(locals(), a="a", b="b"):
        try:
            a = 1
            b = 2
            _scream()
        finally:
            locals()


def test_jetsam_locals_simple(caplog):
    with pytest.raises(Exception, match="ABC") as excinfo:
        _jetsamed_fn()
    assert excinfo.value.graphkit_jetsam == {"a": 1, "b": 2}
    assert "Supressed error" not in caplog.text


def test_jetsam_nested():
    def inner():
        with base.jetsam(locals(), fn="fn"):
            try:
                a = 0
                fn = "inner"
                _jetsamed_fn()
            finally:
                locals()

    def outer():
        with base.jetsam(locals(), fn="fn"):
            try:

                fn = "outer"
                b = 0
                inner()
            finally:
                locals()

    with pytest.raises(Exception, match="ABC") as excinfo:
        outer()

    assert excinfo.value.graphkit_jetsam == {"fn": "inner", "a": 1, "b": 2}


def screaming_dumy_op():
    # No jetsam, in particular, to check sites.
    class Op:
        _compute = _scream

    return Op()


@pytest.mark.parametrize(
    "acallable, expected_jetsam",
    [
        # NO old-stuff Operation(fn=_jetsamed_fn, name="test", needs="['a']", provides=[]),
        (
            fnt.partial(
                operation(name="test", needs=["a"], provides=["b"])(_scream)._compute,
                named_inputs={"a": 1},
            ),
            "outputs provides results operation args".split(),
        ),
        (
            fnt.partial(
                network.ExecutionPlan(*([None] * 7))._call_operation,
                op=screaming_dumy_op(),
                solution={},
            ),
            ["plan"],
        ),
        # Not easy to test Network calling a screaming func (see next TC).
    ],
)
def test_jetsam_sites_screaming_func(acallable, expected_jetsam):
    # Check jetsams when the underlying function fails.
    with pytest.raises(Exception, match="ABC") as excinfo:
        acallable()

    ex = excinfo.value
    assert set(ex.graphkit_jetsam.keys()) == set(expected_jetsam)

@pytest.mark.parametrize(
    "acallable, expected_jetsam",
    [
        # NO old-stuff Operation(fn=_jetsamed_fn, name="test", needs="['a']", provides=[]),
        (
            fnt.partial(
                operation(name="test", needs=["a"], provides=["b"])(_scream)._compute,
                named_inputs=None,
            ),
            "outputs provides results operation args".split(),
        ),
        (
            fnt.partial(
                network.ExecutionPlan(*([None] * 7))._call_operation,
                op=None,
                solution={},
            ),
            ["plan"],
        ),
        (
            fnt.partial(
                network.Network().compute, named_inputs=None, outputs=None
            ),
            "network plan solution outputs".split(),
        ),
    ],
)
def test_jetsam_sites_scream(acallable, expected_jetsam):
    # Check jetsams when the site fails.
    with pytest.raises(Exception) as excinfo:
        acallable()

    ex = excinfo.value
    assert set(ex.graphkit_jetsam.keys()) == set(expected_jetsam)
