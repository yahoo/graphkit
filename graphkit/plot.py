# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

import io
import os

from .base import NetworkOperation, Operation
from .modifiers import optional


def supported_plot_formats():
    """return automatically all `pydot` extensions withlike ``.png``"""
    import pydot

    return [".%s" % f for f in pydot.Dot().formats]


def plot_graph(
    graph,
    filename=None,
    show=False,
    jupyter=False,
    steps=None,
    inputs=None,
    outputs=None,
    solution=None,
):
    """
    Plot a *Graphviz* graph/steps and return it, if no other argument provided.

    Legend:


    NODES:

    - **circle**: function
    - **oval**: subgraph function
    - **house**: given input
    - **inversed-house**: asked output
    - **polygon**: given both as input & asked as output (what?)
    - **square**: intermediate data, neither given nor asked.
    - **red frame**: delete-instruction, to free up memory.
    - **filled**: data node has a value in `solution`, shown in tooltip.
    - **thick frame**: function/data node visited.

    ARROWS

    - **solid black arrows**: dependencies (source-data are``need``\ed
      by target-operations, sources-operations ``provide`` target-data)
    - **dashed black arrows**: optional needs
    - **green-dotted arrows**: execution steps labeled in succession

    :param graph:
        what to plot
    :param str filename:
        Write diagram into a file.
        Common extensions are ``.png .dot .jpg .jpeg .pdf .svg``
        call :func:`network.supported_plot_formats()` for more.
    :param show:
        If it evaluates to true, opens the  diagram in a  matplotlib window.
        If it equals `-1``, it plots but does not open the Window.
    :param jupyter:
        If it evaluates to true, return an SVG suitable to render 
        in *jupyter notebook cells* (`ipython` must be installed).
    :param steps:
        a list of nodes & instructions to overlay on the diagram
    :param inputs:
        an optional name list, any nodes in there are plotted
        as a "house"
    :param outputs:
        an optional name list, any nodes in there are plotted
        as an "inverted-house"
    :param solution:
        an optional dict with values to annotate nodes
        (currently content not shown, but node drawn as "filled")

    :return:
        An instance of the :mod`pydot` graph

    **Example:**

    >>> from graphkit import compose, operation
    >>> from graphkit.modifiers import optional

    >>> pipeline = compose(name="pipeline")(
    ...     operation(name="add", needs=["a", "b1"], provides=["ab1"])(add),
    ...     operation(name="sub", needs=["a", optional("b2")], provides=["ab2"])(lambda a, b=1: a-b),
    ...     operation(name="abb", needs=["ab1", "ab2"], provides=["asked"])(add),
    ... )

    >>> inputs = {'a': 1, 'b1': 2}
    >>> solution=pipeline(inputs)
    >>> pipeline.plot('plot.svg', inputs=inputs, solution=solution, outputs=['asked', 'b1']);

    """
    import pydot

    assert graph is not None

    def get_node_name(a):
        if isinstance(a, Operation):
            return a.name
        return a

    g = pydot.Dot(graph_type="digraph")

    # draw nodes
    for nx_node in graph.nodes:
        kw = {}
        if isinstance(nx_node, str):
            # Only DeleteInstructions data in steps.
            if nx_node in steps:
                kw = {"color": "red", "penwidth": 2}

            # SHAPE change if in inputs/outputs.
            # tip: https://graphviz.gitlab.io/_pages/doc/info/shapes.html
            shape = "rect"
            if inputs and outputs and nx_node in inputs and nx_node in outputs:
                shape = "hexagon"
            else:
                if inputs and nx_node in inputs:
                    shape = "invhouse"
                if outputs and nx_node in outputs:
                    shape = "house"

            # LABEL change from solution.
            if solution and nx_node in solution:
                kw["style"] = "filled"
                kw["fillcolor"] = "gray"
                # kw["tooltip"] = nx_node, solution.get(nx_node)
            node = pydot.Node(name=nx_node, shape=shape, URL="fdgfdf", **kw)
        else:  # Operation
            kw = {}
            shape = "oval" if isinstance(nx_node, NetworkOperation) else "circle"
            if nx_node in steps:
                kw["style"] = "bold"
            node = pydot.Node(name=nx_node.name, shape=shape, **kw)

        g.add_node(node)

    # draw edges
    for src, dst in graph.edges:
        src_name = get_node_name(src)
        dst_name = get_node_name(dst)
        kw = {}
        if isinstance(dst, Operation) and any(
            n == src and isinstance(n, optional) for n in dst.needs
        ):
            kw["style"] = "dashed"
        edge = pydot.Edge(src=src_name, dst=dst_name, **kw)
        g.add_edge(edge)

    # draw steps sequence
    if steps and len(steps) > 1:
        it1 = iter(steps)
        it2 = iter(steps)
        next(it2)
        for i, (src, dst) in enumerate(zip(it1, it2), 1):
            src_name = get_node_name(src)
            dst_name = get_node_name(dst)
            edge = pydot.Edge(
                src=src_name,
                dst=dst_name,
                label=str(i),
                style="dotted",
                color="green",
                fontcolor="green",
                fontname="bold",
                fontsize=18,
                penwidth=3,
                arrowhead="vee",
            )
            g.add_edge(edge)

    # Save plot
    #
    if filename:
        formats = supported_plot_formats()
        _basename, ext = os.path.splitext(filename)
        if not ext.lower() in formats:
            raise ValueError(
                "Unknown file format for saving graph: %s"
                "  File extensions must be one of: %s" % (ext, " ".join(formats))
            )

        g.write(filename, format=ext.lower()[1:])

    ## Return an SVG renderable in jupyter.
    #
    if jupyter:
        from IPython.display import SVG

        g = SVG(data=g.create_svg())

    ## Display graph via matplotlib
    #
    if show:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        png = g.create_png()
        sio = io.BytesIO(png)
        img = mpimg.imread(sio)
        plt.imshow(img, aspect="equal")
        if show != -1:
            plt.show()

    return g
