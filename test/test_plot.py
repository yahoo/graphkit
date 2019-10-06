# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

import sys
from operator import add

import pytest

from graphkit import base, compose, network, operation, plot
from graphkit.modifiers import optional


@pytest.fixture
def pipeline():
    return compose(name="netop")(
        operation(name="add", needs=["a", "b1"], provides=["ab1"])(add),
        operation(name="sub", needs=["a", optional("b2")], provides=["ab2"])(
            lambda a, b=1: a - b
        ),
        operation(name="abb", needs=["ab1", "ab2"], provides=["asked"])(add),
    )


@pytest.fixture(params=[{"a": 1}, {"a": 1, "b1": 2}])
def inputs(request):
    return {"a": 1, "b1": 2}


@pytest.fixture(params=[None, ("a", "b1")])
def input_names(request):
    return request.param


@pytest.fixture(params=[None, ["asked", "b1"]])
def outputs(request):
    return request.param


@pytest.fixture(params=[None, 1])
def solution(pipeline, inputs, outputs, request):
    return request.param and pipeline(inputs, outputs)


###### TEST CASES #######
##


def test_plotting_docstring():
    common_formats = ".png .dot .jpg .jpeg .pdf .svg".split()
    for ext in common_formats:
        assert ext in base.NetworkOperation.plot.__doc__
        assert ext in network.Network.plot.__doc__


def test_plot_formats(pipeline, input_names, outputs, solution, tmp_path):
    ## Generate all formats  (not needing to save files)

    # ...these are not working on my PC, or travis.
    forbidden_formats = ".dia .hpgl .mif .mp .pcl .pic .vtx .xlib".split()
    prev_dot = None
    for ext in plot.supported_plot_formats():
        if ext not in forbidden_formats:
            dot = pipeline.plot(inputs=input_names, outputs=outputs, solution=solution)
            assert dot
            assert ext == ".jpg" or dot != prev_dot
            prev_dot = dot


def test_plot_bad_format(pipeline, tmp_path):
    with pytest.raises(ValueError, match="Unknown file format") as exinfo:
        pipeline.plot(filename="bad.format")

    ## Check help msg lists all siupported formats
    for ext in plot.supported_plot_formats():
        assert exinfo.match(ext)


def test_plot_write_file(pipeline, tmp_path):
    # Try saving a file from one format.

    fpath = tmp_path / "workflow.png"

    dot = pipeline.plot(str(fpath))
    assert fpath.exists()
    assert dot


def test_plot_matplib(pipeline, tmp_path):
    ## Try matplotlib Window, but # without opening a Window.

    if sys.version_info < (3, 5):
        # On PY< 3.5 it fails with:
        #   nose.proxy.TclError: no display name and no $DISPLAY environment variable
        # eg https://travis-ci.org/ankostis/graphkit/jobs/593957996
        import matplotlib

        matplotlib.use("Agg")
    # do not open window in headless travis
    assert pipeline.plot(show=-1)


@pytest.mark.skipif(sys.version_info < (3, 5), reason="ipython-7+ dropped PY3.4-")
def test_plot_jupyter(pipeline, tmp_path):
    ## Try returned  Jupyter SVG.

    dot = pipeline.plot(jupyter=True)
    assert "display.SVG" in str(type(dot))
