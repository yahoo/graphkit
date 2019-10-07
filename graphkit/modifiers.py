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
            operation(name='myadd', needs=['a', 'b', optional('c')], provides='sum')(myadd)
        )

        # The graph works with and without 'c' provided as input.
        assert graph({'a': 5, 'b': 2, 'c': 4})['sum'] == 11
        assert graph({'a': 5, 'b': 2})['sum'] == 7

    """

    pass


class sideffect(str):
    """
    Inputs & outputs in ``needs`` & ``provides`` may be designated as *sideffects*
    using this modifier.  *Tokens* work as usual while solving the DAG but
    they are never assigned any values to/from the ``operation`` functions.
    Specifically:

    - input sideffects are NOT fed into the function;
    - output sideffects are NOT expected from the function.

    Their purpose is to describe functions that have modify internal state
    their arguments ("side-effects").
    Note that an ``operation`` with just a single *sideffect* output return
    no value at all, but it would still be called for its side-effects  only.

    A typical use case is to signify columns required to produce new ones in
    pandas dataframes::

        from graphkit import operation, compose
        from graphkit.modifiers import sideffect

        # Function appending a new dataframe column from two pre-existing ones.
        def addcolumns(df):
            df['sum'] = df['a'] + df['b']

        # Designate `a`, `b` & `sum` column names as an sideffect arguments.
        graph = compose('mygraph')(
            operation(
                name='addcolumns',
                needs=['df', sideffect('a'), sideffect('b')],
                provides=[sideffect('sum')])(addcolumns)
        )

        # The graph works with and without 'c' provided as input.
        df = pd.DataFrame({'a': [5], 'b': [2]})
        assert graph({'df': df})['sum'] == 11

    """

    pass
