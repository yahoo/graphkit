# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

import time
import os
import networkx as nx

from collections import defaultdict
from io import StringIO
from itertools import chain


from boltons.setutils import IndexedSet as iset

from .base import Operation
from .modifiers import optional


class DataPlaceholderNode(str):
    """
    A node for the Network graph that describes the name of a Data instance
    produced or required by a layer.
    """
    def __repr__(self):
        return 'DataPlaceholderNode("%s")' % self


class DeleteInstruction(str):
    """
    An instruction in the *execution plan* to free or delete a Data instance
    from the Network's cache after it is no longer needed.
    """
    def __repr__(self):
        return 'DeleteInstruction("%s")' % self


class Network(object):
    """
    This is the main network implementation. The class contains all of the
    code necessary to weave together operations into a directed-acyclic-graph (DAG)
    and pass data through.

    The computation, ie the execution of the *operations* for given *inputs*
    and asked *outputs* is based on 3 data-structures:

    - The ``networkx`` :attr:`graph` DAG, containing interchanging layers of
      :class:`Operation` and :class:`DataPlaceholderNode` nodes.
      They are layed out and connected by repeated calls of :meth:`add_OP`.

      When the computation starts, :meth:`compile()` extracts a *DAG subgraph*
      by *pruning* nodes based on given inputs and requested outputs.
      This subgraph is used to decide the `execution_plan` (see below), and
      and is cached in :attr:`_cached_execution_plans` across runs with
      thre inputs/outputs as key.

    - the :attr:`execution_plan` lists the operation-nodes & *instructions*
      needed to run a complete  computation.
      It is built in :meth:`_build_execution_plan()` based on the subgraph
      extracted above. The *instructions* items achieve the following:

      - :class:`DeleteInstruction`: delete items from values-cache as soon as
        they are not needed further down the dag, to reduce memory footprint
        while computing.

      - :class:`PinInstruction`: avoid overwritting any given intermediate
        inputs, and still allow their producing operations to run.

    - the :var:`cache` local-var, initialized on each run of both
      ``_compute_xxx`` methods (for parallel or sequential executions), to
      hold all given input & generated (aka intermediate) data values.
    """

    def __init__(self, **kwargs):
        """
        """

        # directed graph of layer instances and data-names defining the net.
        self.graph = nx.DiGraph()
        self._debug = kwargs.get("debug", False)

        # this holds the timing information for eache layer
        self.times = {}

        #: The list of operation-nodes & *instructions* needed to evaluate
        #: the given inputs & asked outputs, free memory and avoid overwritting
        #: any given intermediate inputs.
        self.execution_plan = []

        #: Speed up :meth:`compile()` call and avoid a multithreading issue(?)
        #: that is occuring when accessing the dag in networkx.
        self._cached_execution_plans = {}


    def add_op(self, operation):
        """
        Adds the given operation and its data requirements to the network graph
        based on the name of the operation, the names of the operation's needs,
        and the names of the data it provides.

        :param Operation operation: Operation object to add.
        """

        # assert layer and its data requirements are named.
        assert operation.name, "Operation must be named"
        assert operation.needs is not None, "Operation's 'needs' must be named"
        assert operation.provides is not None, "Operation's 'provides' must be named"

        # assert layer is only added once to graph
        assert operation not in self.graph.nodes(), "Operation may only be added once"

        ## Invalidate old plans.
        self._cached_execution_plans = {}

        # add nodes and edges to graph describing the data needs for this layer
        for n in operation.needs:
            self.graph.add_edge(DataPlaceholderNode(n), operation)

        # add nodes and edges to graph describing what this layer provides
        for p in operation.provides:
            self.graph.add_edge(operation, DataPlaceholderNode(p))


    def list_layers(self, debug=False):
        # Make a generic plan.
        plan = self._build_execution_plan(self.graph)
        return [n for n in plan if debug or isinstance(n, Operation)]


    def show_layers(self, debug=False, ret=False):
        """Shows info (name, needs, and provides) about all operations in this dag."""
        s = "\n".join(repr(n) for n in self.list_layers(debug=debug))
        if ret:
            return s
        else:
            print(s)

    def _build_execution_plan(self, dag):
        """
        Create the list of operation-nodes & *instructions* evaluating all

        operations & instructions needed a) to free memory and b) avoid
        overwritting given intermediate inputs.

        :param dag:
            the original dag but "shrinked", not "broken"

        In the list :class:`DeleteInstructions` steps (DA) are inserted between
        operation nodes to reduce the memory footprint of cached results.
        A DA is inserted whenever a *need* is not used by any other *operation*
        further down the DAG.
        Note that since the *cache* is not reused across `compute()` invocations,
        any memory-reductions are for as long as a single computation runs.

        """

        plan = []

        # create an execution order such that each layer's needs are provided.
        ordered_nodes = iset(nx.topological_sort(dag))

        # add Operations evaluation steps, and instructions to free data.
        for i, node in enumerate(ordered_nodes):

            if isinstance(node, DataPlaceholderNode):
                continue

            elif isinstance(node, Operation):

                plan.append(node)

                # Add instructions to delete predecessors as possible.  A
                # predecessor may be deleted if it is a data placeholder that
                # is no longer needed by future Operations.
                for need in self.graph.pred[node]:
                    if self._debug:
                        print("checking if node %s can be deleted" % need)
                    for future_node in ordered_nodes[i+1:]:
                        if isinstance(future_node, Operation) and need in future_node.needs:
                            break
                    else:
                        if self._debug:
                            print("  adding delete instruction for %s" % need)
                        plan.append(DeleteInstruction(need))

            else:
                raise AssertionError("Unrecognized network graph node %r" % node)

        return plan

    def _collect_unsatisfied_operations(self, dag, inputs):
        """
        Traverse topologically sorted dag to collect un-satisfied operations.

        Unsatisfied operations are those suffering from ANY of the following:

        - They are missing at least one compulsory need-input.
          Since the dag is ordered, as soon as we're on an operation,
          all its needs have been accounted, so we can get its satisfaction.

        - Their provided outputs are not linked to any data in the dag.
          An operation might not have any output link when :meth:`_solve_dag()`
          has broken them, due to given intermediate inputs.

        :param dag:
            the graph to consider
        :param inputs:
            an iterable of the names of the input values
        return:
            a list of unsatisfied operations to prune
        """
        # To collect data that will be produced.
        ok_data = set(inputs)
        # To colect the map of operations --> satisfied-needs.
        op_satisfaction = defaultdict(set)
        # To collect the operations to drop.
        unsatisfied = []
        for node in nx.topological_sort(dag):
            if isinstance(node, Operation):
                if not dag.adj[node]:
                    # Prune operations that ended up providing no output.
                    unsatisfied.append(node)
                else:
                    real_needs = set(n for n in node.needs
                                     if not isinstance(n, optional))
                    if real_needs.issubset(op_satisfaction[node]):
                        # We have a satisfied operation; mark its output-data
                        # as ok.
                        ok_data.update(dag.adj[node])
                    else:
                        # Prune operations with partial inputs.
                        unsatisfied.append(node)
            elif isinstance(node, (DataPlaceholderNode, str)): # `str` are givens
                  if node in ok_data:
                    # mark satisfied-needs on all future operations
                    for future_op in dag.adj[node]:
                        op_satisfaction[future_op].add(node)
            else:
                raise AssertionError("Unrecognized network graph node %r" % node)

        return unsatisfied


    def _solve_dag(self, outputs, inputs):
        """
        Determines what graph steps need to run to get to the requested
        outputs from the provided inputs.  Eliminates steps that come before
        (in topological order) any inputs that have been provided.  Also
        eliminates steps that are not on a path from the provided inputs to
        the requested outputs.

        :param iterable outputs:
            A list of desired output names.  This can also be ``None``, in which
            case the necessary steps are all graph nodes that are reachable
            from one of the provided inputs.

        :param iterable inputs:
            The inputs names of all given inputs.

        :return:
            the *execution plan*
        """
        dag = self.graph

        # Ignore input names that aren't in the graph.
        graph_inputs = iset(dag.nodes) & inputs  # preserve order

        # Scream if some requested outputs aren't in the graph.
        unknown_outputs = iset(outputs) - dag.nodes
        if unknown_outputs:
            raise ValueError(
                "Unknown output node(s) requested: %s"
                % ", ".join(unknown_outputs))

        broken_dag = dag.copy()  # preserve net's graph

        # Break the incoming edges to all given inputs.
        #
        # Nodes producing any given intermediate inputs are unecessary
        # (unless they are also used elsewhere).
        # To discover which ones to prune, we break their incoming edges
        # and they will drop out while collecting ancestors from the outputs.
        for given in graph_inputs:
            broken_dag.remove_edges_from(list(broken_dag.in_edges(given)))

        if outputs:
            # If caller requested specific outputs, we can prune any
            # unrelated nodes further up the dag.
            ending_in_outputs = set()
            for input_name in outputs:
                ending_in_outputs.update(nx.ancestors(dag, input_name))
            broken_dag = broken_dag.subgraph(ending_in_outputs | set(outputs))


        # Prune (un-satifiable) operations with partial inputs.
        # See yahoo/graphkit#18
        #
        unsatisfied = self._collect_unsatisfied_operations(broken_dag, inputs)
        shrinked_dag = dag.subgraph(broken_dag.nodes - unsatisfied)

        plan = self._build_execution_plan(shrinked_dag)

        return plan


    def compile(self, outputs=(), inputs=()):
        """
        Solve dag, set the :attr:`execution_plan`, and cache it.

        See :meth:`_solve_dag()` for detailed description.

        :param iterable outputs:
            A list of desired output names.  This can also be ``None``, in which
            case the necessary steps are all graph nodes that are reachable
            from one of the provided inputs.

        :param dict inputs:
            The input names of all given inputs.
        """

        # return steps if it has already been computed before for this set of inputs and outputs
        if outputs is not None and not isinstance(outputs, str):
            outputs = tuple(sorted(outputs))
        inputs_keys = tuple(sorted(inputs))
        cache_key = (inputs_keys, outputs)
        if cache_key in self._cached_execution_plans:
            self.execution_plan = self._cached_execution_plans[cache_key]
        else:
            plan = self._solve_dag(outputs, inputs)
            # save this result in a precomputed cache for future lookup
            self.execution_plan = self._cached_execution_plans[cache_key] = plan



    def compute(self, outputs, named_inputs, method=None):
        """
        Run the graph. Any inputs to the network must be passed in by name.

        :param list output: The names of the data node you'd like to have returned
                            once all necessary computations are complete.
                            If you set this variable to ``None``, all
                            data nodes will be kept and returned at runtime.

        :param dict named_inputs: A dict of key/value pairs where the keys
                                  represent the data nodes you want to populate,
                                  and the values are the concrete values you
                                  want to set for the data node.

        :param method:
            if ``"parallel"``, launches multi-threading.
            Set when invoking a composed graph or by
            :meth:`~NetworkOperation.set_execution_method()`.

        :returns: a dictionary of output data objects, keyed by name.
        """

        assert isinstance(outputs, (list, tuple)) or outputs is None,\
            "The outputs argument must be a list"

        # start with fresh data cache & overwrites
        cache = named_inputs.copy()

        # Build and set :attr:`execution_plan`.
        self.compile(outputs, named_inputs.keys())

        # choose a method of execution
        if method == "parallel":
            self._compute_thread_pool_barrier_method(cache)
        else:
            self._compute_sequential_method(cache, outputs)

        if not outputs:
            # Return the whole cache as output, including input and
            # intermediate data nodes.
            return cache

        else:
            # Filter outputs to just return what's needed.
            # Note: list comprehensions exist in python 2.7+
            return dict(i for i in cache.items() if i[0] in outputs)


    def _compute_thread_pool_barrier_method(
        self, cache, thread_pool_size=10
    ):
        """
        This method runs the graph using a parallel pool of thread executors.
        You may achieve lower total latency if your graph is sufficiently
        sub divided into operations using this method.
        """
        from multiprocessing.dummy import Pool

        # if we have not already created a thread_pool, create one
        if not hasattr(self, "_thread_pool"):
            self._thread_pool = Pool(thread_pool_size)
        pool = self._thread_pool


        # this keeps track of all nodes that have already executed
        has_executed = set()  # unordered, not iterated

        # with each loop iteration, we determine a set of operations that can be
        # scheduled, then schedule them onto a thread pool, then collect their
        # results onto a memory cache for use upon the next iteration.
        while True:

            # the upnext list contains a list of operations for scheduling
            # in the current round of scheduling
            upnext = []
            for node in self.execution_plan:
                # only delete if all successors for the data node have been executed
                if isinstance(node, DeleteInstruction):
                    if ready_to_delete_data_node(node,
                                                 has_executed,
                                                 self.graph):
                        if node in cache:
                            cache.pop(node)

                # continue if this node is anything but an operation node
                if not isinstance(node, Operation):
                    continue

                if ready_to_schedule_operation(node, has_executed, self.graph) \
                        and node not in has_executed:
                    upnext.append(node)


            # stop if no nodes left to schedule, exit out of the loop
            if len(upnext) == 0:
                break

            done_iterator = pool.imap_unordered(
                                lambda op: (op,op._compute(cache)),
                                upnext)
            for op, result in done_iterator:
                cache.update(result)
                has_executed.add(op)


    def _compute_sequential_method(self, cache, outputs):
        """
        This method runs the graph one operation at a time in a single thread
        """
        self.times = {}
        for step in self.execution_plan:

            if isinstance(step, Operation):

                if self._debug:
                    print("-"*32)
                    print("executing step: %s" % step.name)

                # time execution...
                t0 = time.time()

                # compute layer outputs
                layer_outputs = step._compute(cache)

                # add outputs to cache
                cache.update(layer_outputs)

                # record execution time
                t_complete = round(time.time() - t0, 5)
                self.times[step.name] = t_complete
                if self._debug:
                    print("step completion time: %s" % t_complete)

            # Process DeleteInstructions by deleting the corresponding data
            # if possible.
            elif isinstance(step, DeleteInstruction):

                if outputs and step not in outputs:
                    # Some DeleteInstruction steps may not exist in the cache
                    # if they come from optional() needs that are not privoded
                    # as inputs.  Make sure the step exists before deleting.
                    if step in cache:
                        if self._debug:
                            print("removing data '%s' from cache." % step)
                        cache.pop(step)

            else:
                raise AssertionError("Unrecognized instruction.%r" % step)


    def plot(self, filename=None, show=False):
        """
        Plot the graph.

        params:
        :param str filename:
            Write the output to a png, pdf, or graphviz dot file. The extension
            controls the output format.

        :param boolean show:
            If this is set to True, use matplotlib to show the graph diagram
            (Default: False)

        :returns:
            An instance of the pydot graph

        """
        import pydot
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        assert self.graph is not None

        def get_node_name(a):
            if isinstance(a, DataPlaceholderNode):
                return a
            return a.name

        g = pydot.Dot(graph_type="digraph")

        # draw nodes
        for nx_node in self.graph.nodes():
            if isinstance(nx_node, DataPlaceholderNode):
                node = pydot.Node(name=nx_node, shape="rect")
            else:
                node = pydot.Node(name=nx_node.name, shape="circle")
            g.add_node(node)

        # draw edges
        for src, dst in self.graph.edges():
            src_name = get_node_name(src)
            dst_name = get_node_name(dst)
            edge = pydot.Edge(src=src_name, dst=dst_name)
            g.add_edge(edge)

        # save plot
        if filename:
            _basename, ext = os.path.splitext(filename)
            with open(filename, "wb") as fh:
                if ext.lower() == ".png":
                    fh.write(g.create_png())
                elif ext.lower() == ".dot":
                    fh.write(g.to_string())
                elif ext.lower() in [".jpg", ".jpeg"]:
                    fh.write(g.create_jpeg())
                elif ext.lower() == ".pdf":
                    fh.write(g.create_pdf())
                elif ext.lower() == ".svg":
                    fh.write(g.create_svg())
                else:
                    raise Exception("Unknown file format for saving graph: %s" % ext)

        # display graph via matplotlib
        if show:
            png = g.create_png()
            sio = StringIO(png)
            img = mpimg.imread(sio)
            plt.imshow(img, aspect="equal")
            plt.show()

        return g


def ready_to_schedule_operation(op, has_executed, graph):
    """
    Determines if a Operation is ready to be scheduled for execution based on
    what has already been executed.

    Args:
        op:
            The Operation object to check
        has_executed: set
            A set containing all operations that have been executed so far
        graph:
            The networkx graph containing the operations and data nodes
    Returns:
        A boolean indicating whether the operation may be scheduled for
        execution based on what has already been executed.
    """
    # unordered, not iterated
    dependencies = set(filter(lambda v: isinstance(v, Operation),
                              nx.ancestors(graph, op)))
    return dependencies.issubset(has_executed)

def ready_to_delete_data_node(name, has_executed, graph):
    """
    Determines if a DataPlaceholderNode is ready to be deleted from the
    cache.

    Args:
        name:
            The name of the data node to check
        has_executed: set
            A set containing all operations that have been executed so far
        graph:
            The networkx graph containing the operations and data nodes
    Returns:
        A boolean indicating whether the data node can be deleted or not.
    """
    data_node = get_data_node(name, graph)
    return set(graph.successors(data_node)).issubset(has_executed)

def get_data_node(name, graph):
    """
    Gets a data node from a graph using its name
    """
    for node in graph.nodes():
        if node == name and isinstance(node, DataPlaceholderNode):
            return node
    return None
