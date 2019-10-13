# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import logging

import pytest
import itertools as itt

from graphkit import base


def test_jetsam_without_failure(caplog):
    caplog.set_level(logging.INFO)
    with pytest.raises(AssertionError, match="No `keys_to_salvage`"):
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
    with pytest.raises(AssertionError, match="Bad `keys_to_salvage`") as excinfo:
        with base.jetsam({}, **keys):
            raise Exception("ABC")

    assert not hasattr(excinfo.value, "graphkit_jetsam")
    assert "Supressed error while annotating exception" not in caplog.text


@pytest.mark.parametrize("locs", [None, (), [], [0], "bad"])
def test_jetsam_bad_locals_given(locs, caplog):
    caplog.set_level(logging.INFO)
    with pytest.raises(AssertionError, match="`locs` given to jetsam") as excinfo:
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


def _jetsamed_fn():
    b = 1
    with base.jetsam(locals(), a="a", b="b"):
        try:
            a = 1
            b = 2
            raise Exception("ABC", a, b)
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

                fn = "inner"
                raise Exception("ABC")
            finally:
                locals()

    def outer():
        with base.jetsam(locals(), fn="fn"):
            try:

                fn = "outer"
                inner()
            finally:
                locals()

    with pytest.raises(Exception, match="ABC") as excinfo:
        outer()

    assert excinfo.value.graphkit_jetsam == {"fn": "inner"}
