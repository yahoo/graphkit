# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

import io
import logging
import os


log = logging.getLogger(__name__)


class Plotter(object):
    """
    Classes wishing to plot their graphs should inherit this and ...

    implement property ``_plot`` to return a "partial" callable that somehow
    ends up calling  :func:`plot.plot_graph()` with the `graph` or any other
    args binded appropriately.
    The purpose is to avoid copying this function & documentation here around.
    """

    def plot(self, filename=None, show=False, **kws):
        """
        :param str filename:
            Write diagram into a file.
            Common extensions are ``.png .dot .jpg .jpeg .pdf .svg``
            call :func:`plot.supported_plot_formats()` for more.
        :param show:
            If it evaluates to true, opens the  diagram in a  matplotlib window.
            If it equals `-1`, it plots but does not open the Window.
        :param inputs:
            an optional name list, any nodes in there are plotted
            as a "house"
        :param outputs:
            an optional name list, any nodes in there are plotted
            as an "inverted-house"
        :param solution:
            an optional dict with values to annotate nodes, drawn "filled"
            (currently content not shown, but node drawn as "filled")
        :param executed:
            an optional container with operations executed, drawn "filled"
        :param title:
            an optional string to display at the bottom of the graph
        :param node_props:
            an optional nested dict of Grapvhiz attributes for certain nodes
        :param edge_props:
            an optional nested dict of Grapvhiz attributes for certain edges
        :param clusters:
            an optional mapping of nodes --> cluster-names, to group them

        :return:
            A ``pydot.Dot`` instance.
            NOTE that the returned instance is monkeypatched to support
            direct rendering in *jupyter cells* as SVG.


        Note that the `graph` argument is absent - Each Plotter provides
        its own graph internally;  use directly :func:`plot_graph()` to provide
        a different graph.

        **Legend:**

        . figure:: ../images/Graphkitlegend.svg
            :alt: Graphkit Legend

            see :func:`legend()`

        *NODES:*

        oval
            function
        egg
            subgraph operation
        house
            given input
        inversed-house
            asked output
        polygon
            given both as input & asked as output (what?)
        square
            intermediate data, neither given nor asked.
        red frame
            delete-instruction, to free up memory.
        blue frame
            pinned-instruction, not to overwrite intermediate inputs.
        filled
            data node has a value in `solution` OR function has been executed.
        thick frame
            function/data node in execution `steps`.

        *ARROWS*

        solid black arrows
            dependencies (source-data are``need``-ed by target-operations,
            sources-operations ``provide`` target-data)
        dashed black arrows
            optional needs
        wheat arrows
            broken dependency (``provide``) during pruning
        green-dotted arrows
            execution steps labeled in succession

        **Sample code:**

        >>> from graphkit import compose, operation
        >>> from graphkit.modifiers import optional

        >>> pipeline = compose(name="pipeline")(
        ...     operation(name="add", needs=["a", "b1"], provides=["ab1"])(add),
        ...     operation(name="sub", needs=["a", optional("b2")], provides=["ab2"])(lambda a, b=1: a-b),
        ...     operation(name="abb", needs=["ab1", "ab2"], provides=["asked"])(add),
        ... )

        >>> inputs = {'a': 1, 'b1': 2}
        >>> solution=pipeline(inputs)

        >>> pipeline.plot('plot1.svg', inputs=inputs, outputs=['asked', 'b1'], solution=solution);
        >>> pipeline.last_plan.plot('plot2.svg', solution=solution);
        """
        dot = self._build_pydot(**kws)
        return render_pydot(dot, filename=filename, show=show)

    def _build_pydot(self, **kws):
        raise AssertionError("Must implement that!")


def _is_class_value_in_list(lst, cls, value):
    return any(isinstance(i, cls) and i == value for i in lst)


def _merge_conditions(*conds):
    """combines conditions as a choice in binary range, eg, 2 conds --> [0, 3]"""
    return sum(int(bool(c)) << i for i, c in enumerate(conds))


def _apply_user_props(dotobj, user_props, key):
    if user_props and key in user_props:
        dotobj.get_attributes().update(user_props[key])
        # Delete it, to report unmatched ones, AND not to annotate `steps`.
        del user_props[key]


def _report_unmatched_user_props(user_props, kind):
    if user_props and log.isEnabledFor(logging.WARNING):
        unmatched = "\n  ".join(str(i) for i in user_props.items())
        log.warning("Unmatched `%s_props`:\n  +--%s", kind, unmatched)


def _monkey_patch_for_jupyter(pydot):
    # Ensure Dot nstance render in Jupyter
    # (see pydot/pydot#220)
    if not hasattr(pydot.Dot, "_repr_svg_"):

        def make_svg(self):
            return self.create_svg().decode()

        # monkey patch class
        pydot.Dot._repr_svg_ = make_svg


def build_pydot(
    graph,
    steps=None,
    inputs=None,
    outputs=None,
    solution=None,
    executed=None,
    title=None,
    node_props=None,
    edge_props=None,
    clusters=None,
):
    """
    Build a *Graphviz* out of a Network graph/steps/inputs/outputs and return it.

    See :meth:`Plotter.plot()` for the arguments, sample code, and
    the legend of the plots.
    """
    import pydot
    from .base import NetworkOperation, Operation
    from .modifiers import optional
    from .network import DeleteInstruction, PinInstruction

    _monkey_patch_for_jupyter(pydot)

    assert graph is not None

    steps_thickness = 3
    fill_color = "wheat"
    steps_color = "#009999"
    new_clusters = {}

    def append_or_cluster_node(dot, nx_node, node):
        if not clusters or not nx_node in clusters:
            dot.add_node(node)
        else:
            cluster_name = clusters[nx_node]
            node_cluster = new_clusters.get(cluster_name)
            if not node_cluster:
                node_cluster = new_clusters[cluster_name] = pydot.Cluster(
                    cluster_name, label=cluster_name
                )
            node_cluster.add_node(node)

    def append_any_clusters(dot):
        for cluster in new_clusters.values():
            dot.add_subgraph(cluster)

    def get_node_name(a):
        if isinstance(a, Operation):
            return a.name
        return a

    dot = pydot.Dot(graph_type="digraph", label=title, fontname="italic")

    # draw nodes
    for nx_node in graph.nodes:
        if isinstance(nx_node, str):
            kw = {}

            # FrameColor change by step type
            if steps and nx_node in steps:
                choice = _merge_conditions(
                    _is_class_value_in_list(steps, DeleteInstruction, nx_node),
                    _is_class_value_in_list(steps, PinInstruction, nx_node),
                )
                # 0 is singled out because `nx_node` exists in `steps`.
                color = "NOPE #990000 blue purple".split()[choice]
                kw = {"color": color, "penwidth": steps_thickness}

            # SHAPE change if with inputs/outputs.
            # tip: https://graphviz.gitlab.io/_pages/doc/info/shapes.html
            choice = _merge_conditions(
                inputs and nx_node in inputs, outputs and nx_node in outputs
            )
            shape = "rect invhouse house hexagon".split()[choice]

            # LABEL change with solution.
            if solution and nx_node in solution:
                kw["style"] = "filled"
                kw["fillcolor"] = fill_color
                # kw["tooltip"] = str(solution.get(nx_node))  # not working :-()
            node = pydot.Node(name=nx_node, shape=shape, **kw)
        else:  # Operation
            kw = {}

            if steps and nx_node in steps:
                kw["penwdth"] = steps_thickness
            shape = "egg" if isinstance(nx_node, NetworkOperation) else "oval"
            if executed and nx_node in executed:
                kw["style"] = "filled"
                kw["fillcolor"] = fill_color
            node = pydot.Node(name=nx_node.name, shape=shape, **kw)

        _apply_user_props(node, node_props, key=node.get_name())

        append_or_cluster_node(dot, nx_node, node)

    _report_unmatched_user_props(node_props, "node")

    append_any_clusters(dot)

    # draw edges
    for src, dst in graph.edges:
        src_name = get_node_name(src)
        dst_name = get_node_name(dst)
        kw = {}
        if isinstance(dst, Operation) and _is_class_value_in_list(
            dst.needs, optional, src
        ):
            kw["style"] = "dashed"
        edge = pydot.Edge(src=src_name, dst=dst_name, **kw)

        _apply_user_props(edge, edge_props, key=(src, dst))

        dot.add_edge(edge)

    _report_unmatched_user_props(edge_props, "edge")

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
                color=steps_color,
                fontcolor=steps_color,
                fontname="bold",
                fontsize=18,
                penwidth=steps_thickness,
                arrowhead="vee",
            )
            dot.add_edge(edge)

    return dot


def supported_plot_formats():
    """return automatically all `pydot` extensions withlike ``.png``"""
    import pydot

    return [".%s" % f for f in pydot.Dot().formats]


def render_pydot(dot, filename=None, show=False):
    """
    Plot a *Graphviz* dot in a matplotlib, in file or return it for Jupyter.

    :param dot:
        the pre-built *Graphviz* dot instance
    :param str filename:
        Write diagram into a file.
        Common extensions are ``.png .dot .jpg .jpeg .pdf .svg``
        call :func:`plot.supported_plot_formats()` for more.
    :param show:
        If it evaluates to true, opens the  diagram in a  matplotlib window.
        If it equals `-1`, it returns the image but does not open the Window.

    :return:
        the matplotlib image if ``show=-1``, or the `dot`.

    See :meth:`Plotter.plot()` for sample code.
    """
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

        dot.write(filename, format=ext.lower()[1:])

    ## Display graph via matplotlib
    #
    if show:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        png = dot.create_png()
        sio = io.BytesIO(png)
        img = mpimg.imread(sio)
        if show != -1:
            plt.imshow(img, aspect="equal")
            plt.show()

        return img

    return dot


def legend(filename=None, show=None):
    """Generate a legend for all plots (see Plotter.plot() for args)"""
    import pydot

    _monkey_patch_for_jupyter(pydot)

    ## From https://stackoverflow.com/questions/3499056/making-a-legend-key-in-graphviz
    dot_text = """
    digraph {
        rankdir=LR;
        subgraph cluster_legend {
        label="Graphkit Legend";

        operation   [shape=oval];
        graphop     [shape=egg label="graph operation"];
        insteps     [penwidth=3 label="execution step"];
        executed    [style=filled fillcolor=wheat];
        operation -> graphop -> insteps -> executed [style=invis];

        data    [shape=rect];
        input   [shape=invhouse];
        output  [shape=house];
        inp_out [shape=hexagon label="inp+out"];
        evicted [shape=rect penwidth=3 color="#990000"];
        pinned  [shape=rect penwidth=3 color="blue"];
        evpin   [shape=rect penwidth=3 color=purple label="evict+pin"];
        sol     [shape=rect style=filled fillcolor=wheat label="in solution"];
        data -> input -> output -> inp_out -> evicted -> pinned -> evpin -> sol [style=invis];

        e1 [style=invis] e2 [color=invis label="dependency"];
        e1 -> e2;
        e3 [color=invis label="optional"];
        e2 -> e3 [style=dashed];
        e4 [color=invis penwidth=3 label="pruned dependency"];
        e3 -> e4 [color=wheat penwidth=2];
        e5 [color=invis penwidth=4 label="execution sequence"];
        e4 -> e5 [color="#009999" penwidth=4 style=dotted arrowhead=vee label=1 fontcolor="#009999"];
        }
    }
    """

    dot = pydot.graph_from_dot_data(dot_text)[0]
    # clus = pydot.Cluster("Graphkit legend", label="Graphkit legend")
    # dot.add_subgraph(clus)

    # nodes = dot.Node()
    # clus.add_node("operation")

    return render_pydot(dot, filename=filename, show=show)
