# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""" The main implementation of the network of operations & data to compute. """
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
    Dag node naming a data-value produced or required by an operation.
    """
    def __repr__(self):
        return 'DataPlaceholderNode("%s")' % self


class DeleteInstruction(str):
    """
    Execution step to delete a computed value from the network's ``cache``.

    It is an :attr:`Network.execution_plan` step for the data-node `str` that
    frees its data-value from ``cache`` after it is no longer needed,
    to reduce memory footprint while computing the pipeline.
    """
    def __repr__(self):
        return 'DeleteInstruction("%s")' % self


class PinInstruction(str):
    """
    Execution step to replace a computed value in the ``cache`` from the inputs,

    and to store the computed one in the ``overwrites`` instead
    (both ``cache`` & ``overwrites`` are local-vars in :meth:`Network.compute()`).

    It is an :attr:`Network.execution_plan` step for the data-node `str` that
    ensures the corresponding intermediate input-value is not overwritten when
    its providing function(s) could not be pruned, because their other outputs
    are needed elesewhere.
    """
    def __repr__(self):
        return 'PinInstruction("%s")' % self


class Network(object):
    """
    Assemble operations & data into a directed-acyclic-graph (DAG) to run them.

    The execution of the contained *operations* in the dag (the computation)
    is splitted in 2 phases:

    - COMPILE: prune unsatisfied nodes, sort dag topologically & solve it, and
      derive the *execution plan* (see below) based on the given *inputs*
      and asked *outputs*.

    - EXECUTE: sequential or parallel invocation of the underlying functions
      of the operations with arguments from the ``cache``.

    is based on 5 data-structures:

    :ivar graph:
        A ``networkx`` DAG containing interchanging layers of
        :class:`Operation` and :class:`DataPlaceholderNode` nodes.
        They are layed out and connected by repeated calls of :meth:`add_OP`.

        The computation starts with :meth:`_prune_dag()` extracting
        a *DAG subgraph* by *pruning* its nodes based on given inputs and
        requested outputs in :meth:`compute()`.
    :ivar execution_dag:
        It contains the nodes of the *pruned dag* from the last call to
        :meth:`compile()`. This pruned subgraph is used to decide
        the :attr:`execution_plan` (below).
        It is cached in :attr:`_cached_compilations` across runs with
        inputs/outputs as key.

    :ivar execution_plan:
        It is the list of the operation-nodes only
        from the dag (above), topologically sorted, and interspersed with
        *instructions steps* needed to complete the run.
        It is built by :meth:`_build_execution_plan()` based on the subgraph dag
        extracted above.
        It is cached in :attr:`_cached_compilations` across runs with
        inputs/outputs as key.

        The *instructions* items achieve the following:

        - :class:`DeleteInstruction`: delete items from values-cache as soon as
          they are not needed further down the dag, to reduce memory footprint
          while computing.

        - :class:`PinInstruction`: avoid overwritting any given intermediate
          inputs, and still allow their providing operations to run
          (because they are needed for their other outputs).

    :var cache:
        a local-var in :meth:`compute()`, initialized on each run
        to hold the values of the given inputs, generated (intermediate) data,
        and output values.
        It is returned as is if no specific outputs requested;  no data-eviction
        happens then.

    :arg overwrites:
        The optional argument given to :meth:`compute()` to colect the
        intermediate *calculated* values that are overwritten by intermediate
        (aka "pinned") input-values.

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
        self.execution_plan = ()

        #: Pruned graph of the last compilation.
        self.execution_dag = ()

        #: Speed up :meth:`compile()` call and avoid a multithreading issue(?)
        #: that is occuring when accessing the dag in networkx.
        self._cached_compilations = {}


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
        assert operation not in self.graph.nodes, "Operation may only be added once"

        self.execution_dag = None
        self.execution_plan = None
        self._cached_compilations = {}

        # add nodes and edges to graph describing the data needs for this layer
        for n in operation.needs:
            self.graph.add_edge(DataPlaceholderNode(n), operation)

        # add nodes and edges to graph describing what this layer provides
        for p in operation.provides:
            self.graph.add_edge(operation, DataPlaceholderNode(p))


    def list_layers(self, debug=False):
        # Make a generic plan.
        plan = self._build_execution_plan(self.graph, ())
        return [n for n in plan if debug or isinstance(n, Operation)]


    def show_layers(self, debug=False, ret=False):
        """Shows info (name, needs, and provides) about all operations in this dag."""
        s = "\n".join(repr(n) for n in self.list_layers(debug=debug))
        if ret:
            return s
        else:
            print(s)

    def _build_execution_plan(self, dag, inputs, outputs):
        """
        Create the list of operation-nodes & *instructions* evaluating all

        operations & instructions needed a) to free memory and b) avoid
        overwritting given intermediate inputs.

        :param dag:
            The original dag, pruned; not broken.
        :param outputs:
            outp-names to decide whether to add (and which) del-instructions

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

        # Add Operations evaluation steps, and instructions to free and "pin"
        # data.
        for i, node in enumerate(ordered_nodes):

            if isinstance(node, DataPlaceholderNode):
                if node in inputs and dag.pred[node]:
                    # Command pinning only when there is another operation
                    # generating this data as output.
                    plan.append(PinInstruction(node))

            elif isinstance(node, Operation):
                plan.append(node)

                # Keep all values in cache if not specific outputs asked.
                if not outputs:
                    continue

                # Add instructions to delete predecessors as possible.  A
                # predecessor may be deleted if it is a data placeholder that
                # is no longer needed by future Operations.
                for need in self.graph.pred[node]:
                    if self._debug:
                        print("checking if node %s can be deleted" % need)
                    for future_node in ordered_nodes[i+1:]:
                        if (
                            isinstance(future_node, Operation)
                            and need in future_node.needs
                        ):
                            break
                    else:
                        if need not in outputs:
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
          An operation might not have any output link when :meth:`_prune_dag()`
          has broken them, due to given intermediate inputs.

        :param dag:
            a graph with broken edges those arriving to existing inputs
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


    def _prune_dag(self, outputs, inputs):
        """
        Determines what graph steps need to run to get to the requested
        outputs from the provided inputs. :
        - Eliminate steps that are not on a path arriving to requested outputs.
        - Eliminate unsatisfied operations: partial inputs or no outputs needed.

        :param iterable outputs:
            A list of desired output names.  This can also be ``None``, in which
            case the necessary steps are all graph nodes that are reachable
            from one of the provided inputs.

        :param iterable inputs:
            The inputs names of all given inputs.

        :return:
            the *pruned_dag*
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


        # Prune unsatisfied operations (those with partial inputs or no outputs).
        unsatisfied = self._collect_unsatisfied_operations(broken_dag, inputs)
        # Clone it so that it is picklable.
        pruned_dag = dag.subgraph(broken_dag.nodes - unsatisfied)

        return pruned_dag.copy()  # clone so that it is picklable


    def compile(self, outputs=(), inputs=()):
        """
        Solve dag, set the :attr:`execution_plan`, and cache it.

        See :meth:`_prune_dag()` for detailed description.

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

        if cache_key in self._cached_compilations:
            dag, plan = self._cached_compilations[cache_key]
        else:
            dag = self._prune_dag(outputs, inputs)
            plan = self._build_execution_plan(dag, inputs, outputs)

            # Cache compilation results to speed up future runs
            # with different values (but same number of inputs/outputs).
            self._cached_compilations[cache_key] = dag, plan

        ## TODO: Extract into Solution class
        self.execution_dag = dag
        self.execution_plan = plan



    def compute(
        self, outputs, named_inputs, method=None, overwrites_collector=None):
        """
        Solve & execute the graph, sequentially or parallel.

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

        :param overwrites_collector:
            (optional) a mutable dict to be fillwed with named values.
            If missing, values are simply discarded.

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
            self._execute_thread_pool_barrier_method(
                cache, overwrites_collector, named_inputs)
        else:
            self._execute_sequential_method(
                cache, overwrites_collector, named_inputs)

        if not outputs:
            # Return the whole cache as output, including input and
            # intermediate data nodes.
            result = cache

        else:
            # Filter outputs to just return what's needed.
            # Note: list comprehensions exist in python 2.7+
            result = dict(i for i in cache.items() if i[0] in outputs)

        return result


    def _pin_data_in_cache(self, value_name, cache, inputs, overwrites):
        value_name = str(value_name)
        if overwrites is not None:
            overwrites[value_name] = cache[value_name]
        cache[value_name] = inputs[value_name]


    def _execute_thread_pool_barrier_method(
        self, cache, overwrites, inputs, thread_pool_size=10
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
        executed_nodes = set()  # unordered, not iterated

        # with each loop iteration, we determine a set of operations that can be
        # scheduled, then schedule them onto a thread pool, then collect their
        # results onto a memory cache for use upon the next iteration.
        while True:

            # the upnext list contains a list of operations for scheduling
            # in the current round of scheduling
            upnext = []
            for node in self.execution_plan:
                if (
                    isinstance(node, Operation)
                    and self._can_schedule_operation(node, executed_nodes)
                    and node not in executed_nodes
                ):
                    upnext.append(node)
                elif isinstance(node, DeleteInstruction):
                    # Only delete if all successors for the data node
                    # have been executed.
                    # An optional need may not have a value in the cache.
                    if (
                        node in cache
                        and self._can_evict_value(node, executed_nodes)
                    ):
                        if self._debug:
                            print("removing data '%s' from cache." % node)
                        del cache[node]
                elif isinstance(node, PinInstruction):
                    # Always and repeatedely pin the value, even if not all
                    # providers of the data have executed.
                    # An optional need may not have a value in the cache.
                    if node in cache:
                        self._pin_data_in_cache(node, cache, inputs, overwrites)


            # stop if no nodes left to schedule, exit out of the loop
            if len(upnext) == 0:
                break

            done_iterator = pool.imap_unordered(
                                lambda op: (op,op._compute(cache)),
                                upnext)
            for op, result in done_iterator:
                cache.update(result)
                executed_nodes.add(op)


    def _execute_sequential_method(self, cache, overwrites, inputs):
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

            elif isinstance(step, DeleteInstruction):
                # Cache value may be missing if it is optional.
                if step in cache:
                    if self._debug:
                        print("removing data '%s' from cache." % step)
                    del cache[step]

            elif isinstance(step, PinInstruction):
                self._pin_data_in_cache(step, cache, inputs, overwrites)
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
        for nx_node in self.graph.nodes:
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


    def _can_schedule_operation(self, op, executed_nodes):
        """
        Determines if a Operation is ready to be scheduled for execution

        based on what has already been executed.

        :param op:
            The Operation object to check
        :param set executed_nodes
            A set containing all operations that have been executed so far
        :return:
            A boolean indicating whether the operation may be scheduled for
            execution based on what has already been executed.
        """
        # unordered, not iterated
        dependencies = set(n for n in nx.ancestors(self.execution_dag, op)
                           if isinstance(n, Operation))
        return dependencies.issubset(executed_nodes)


    def _can_evict_value(self, name, executed_nodes):
        """
        Determines if a DataPlaceholderNode is ready to be deleted from cache.

        :param name:
            The name of the data node to check
        :param executed_nodes: set
            A set containing all operations that have been executed so far
        :return:
            A boolean indicating whether the data node can be deleted or not.
        """
        data_node = self.get_data_node(name)
        return data_node and set(
            self.execution_dag.successors(data_node)).issubset(executed_nodes)

    def get_data_node(self, name):
        """
        Retuen the data node from a graph using its name, or None.
        """
        node = self.graph.nodes[name]
        if isinstance(node, DataPlaceholderNode):
            return node
