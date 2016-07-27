"""
This sub-module contains input/output modifiers that can be applied to
arguments to ``needs`` and ``provides`` to let GraphKit know it should treat
them differently.

Copyright 2016, Yahoo Inc.
Licensed under the terms of the Apache License, Version 2.0. See the LICENSE
file associated with the project for terms.
"""

class optional(str):
    """
    Input values in ``needs`` may be designated as optional using this modifier.
    If this modifier is applied to an input value, that value will be input to
    the ``operation`` if it is available.  The function underlying the
    ``operation`` should have a parameter with the same name as the input value
    in ``needs``, and the input value will be passed as a keyword argument if
    it is available.

    Here is an example of an operation that uses an optional argument::

        from graphkit import operation, compose
        from graphkit.modifiers import optional

        # Function that adds either two or three numbers.
        def myadd(a, b, c=0):
            return a + b + c

        # Designate c as an optional argument.
        graph = compose('mygraph')(
            operator(name='myadd', needs=['a', 'b', optional('c')], provides='sum')(myadd)
        )

        # The graph works with and without 'c' provided as input.
        assert graph({'a': 5, 'b': 2, 'c': 4})['sum'] == 11
        assert graph({'a': 5, 'b': 2})['sum'] == 7

    """
    pass
