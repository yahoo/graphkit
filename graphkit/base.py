# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

class Data(object):
    """
    This wraps any data that is consumed or produced
    by a Operation. This data should also know how to serialize
    itself appropriately.

    This class an "abstract" class that should be extended by
    any class working with data in the HiC framework.
    """
    def __init__(self, **kwargs):
        pass

    def get_data(self):
        raise NotImplementedError

    def set_data(self, data):
        raise NotImplementedError

class Operation(object):
    """
    This is an abstract class representing a data transformation. To use this,
    please inherit from this class and customize the ``.compute`` method to your
    specific application.
    """

    def __init__(self, **kwargs):
        """
        Create a new layer instance.
        Names may be given to this layer and its inputs and outputs. This is
        important when connecting layers and data in a Network object, as the
        names are used to construct the graph.

        :param str name: The name the operation (e.g. conv1, conv2, etc..)

        :param list needs: Names of input data objects this layer requires.

        :param list provides: Names of output data objects this provides.

        :param dict params: A dict of key/value pairs representing parameters
                            associated with your operation. These values will be
                            accessible using the ``.params`` attribute of your object.
                            NOTE: It's important that any values stored in this
                            argument must be pickelable.
        """

        # (Optional) names for this layer, and the data it needs and provides
        self.name = kwargs.get('name')
        self.needs = kwargs.get('needs')
        self.provides = kwargs.get('provides')
        self.params = kwargs.get('params', {})

        # call _after_init as final step of initialization
        self._after_init()

    def __eq__(self, other):
        """
        Operation equality is based on name of layer.
        (__eq__ and __hash__ must be overridden together)
        """
        return bool(self.name is not None and
                    self.name == getattr(other, 'name', None))

    def __hash__(self):
        """
        Operation equality is based on name of layer.
        (__eq__ and __hash__ must be overridden together)
        """
        return hash(self.name)

    def compute(self, inputs):
        """
        This method must be implemented to perform this layer's feed-forward
        computation on a given set of inputs.
        :param list inputs:
            A list of :class:`Data` objects on which to run the layer's
            feed-forward computation.
        :returns list:
            Should return a list of :class:`Data` objects representing
            the results of running the feed-forward computation on
            ``inputs``.
        """

        raise NotImplementedError

    def _compute(self, named_inputs, outputs=None):
        inputs = [named_inputs[d] for d in self.needs]
        results = self.compute(inputs)

        results = zip(self.provides, results)
        if outputs:
            outputs = set(outputs)
            results = filter(lambda x: x[0] in outputs, results)

        return dict(results)

    def _after_init(self):
        """
        This method is a hook for you to override. It gets called after this
        object has been initialized with its ``needs``, ``provides``, ``name``,
        and ``params`` attributes. People often override this method to implement
        custom loading logic required for objects that do not pickle easily, and
        for initialization of c++ dependencies.
        """
        pass

    def __getstate__(self):
        """
        This allows your operation to be pickled.
        Everything needed to instantiate your operation should be defined by the
        following attributes: params, needs, provides, and name
        No other piece of state should leak outside of these 4 variables
        """

        result = {}
        # this check should get deprecated soon. its for downward compatibility
        # with earlier pickled operation objects
        if hasattr(self, 'params'):
            result["params"] = self.__dict__['params']
        result["needs"] = self.__dict__['needs']
        result["provides"] = self.__dict__['provides']
        result["name"] = self.__dict__['name']

        return result

    def __setstate__(self, state):
        """
        load from pickle and instantiate the detector
        """
        for k in iter(state):
            self.__setattr__(k, state[k])
        self._after_init()

    def __repr__(self):
        """
        Display more informative names for the Operation class
        """
        return u"%s(name='%s', needs=%s, provides=%s)" % \
            (self.__class__.__name__,
             self.name,
             self.needs,
             self.provides)


class NetworkOperation(Operation):
    def __init__(self, **kwargs):
        self.net = kwargs.pop('net')
        Operation.__init__(self, **kwargs)

        # set execution mode to single-threaded sequential by default
        self._execution_method = "sequential"

    def _compute(self, named_inputs, outputs=None):
        return self.net.compute(outputs, named_inputs, method=self._execution_method)

    def __call__(self, *args, **kwargs):
        return self._compute(*args, **kwargs)

    def set_execution_method(self, method):
        """
        Determine how the network will be executed.

        Args:
            method: str
                If "parallel", execute graph operations concurrently
                using a threadpool.
        """
        options = ['parallel', 'sequential']
        assert method in options
        self._execution_method = method

    def plot(self, filename=None, show=False):
        self.net.plot(filename=filename, show=show)

    def __getstate__(self):
        state = Operation.__getstate__(self)
        state['net'] = self.__dict__['net']
        return state
