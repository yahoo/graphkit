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
    An optional need signifies that the function's argument may not receive a value.

    Only input values in ``needs`` may be designated as optional using this modifier.
    An ``operation`` will receive a value for an optional need only if if it is available
    in the graph at the time of its invocation.
    The ``operation``'s function should have a defaulted parameter with the same name
    as the opetional, and the input value will be passed as a keyword argument,
    if it is available.

    Here is an example of an operation that uses an optional argument::

        >>> from graphkit import operation, compose, optional

        >>> # Function that adds either two or three numbers.
        >>> def myadd(a, b, c=0):
        ...    return a + b + c

        >>> # Designate c as an optional argument.
        >>> graph = compose('mygraph')(
        ...     operation(name='myadd', needs=['a', 'b', optional('c')], provides='sum')(myadd)
        ... )
        >>> graph
        NetworkOperation(name='mygraph',
                         needs=[optional('a'), optional('b'), optional('c')],
                         provides=['sum'])

        >>> # The graph works with and without 'c' provided as input.
        >>> graph({'a': 5, 'b': 2, 'c': 4})['sum']
        11
        >>> graph({'a': 5, 'b': 2})
        {'a': 5, 'b': 2, 'sum': 7}

    """

    __slots__ = ()  # avoid __dict__ on instances

    def __repr__(self):
        return "optional('%s')" % self


class sideffect(str):
    """
    A sideffect data-dependency participates in the graph but never given/asked in functions.

    Both inputs & outputs in ``needs`` & ``provides`` may be designated as *sideffects*
    using this modifier.  *Sideffects* work as usual while solving the graph but
    they do not interact with the ``operation``'s function;  specifically:

    - input sideffects are NOT fed into the function;
    - output sideffects are NOT expected from the function.

    .. info:
        an ``operation`` with just a single *sideffect* output return no value at all,
        but it would still be called for its side-effect only.

    Their purpose is to describe operations that modify the internal state of
    some of their arguments ("side-effects").
    A typical use case is to signify columns required to produce new ones in
    pandas dataframes::


        >>> from graphkit import operation, compose, sideffect

        >>> # Function appending a new dataframe column from two pre-existing ones.
        >>> def addcolumns(df):
        ...    df['sum'] = df['a'] + df['b']

        >>> # Designate `a`, `b` & `sum` column names as an sideffect arguments.
        >>> graph = compose('mygraph')(
        ...     operation(
        ...         name='addcolumns',
        ...         needs=['df', sideffect('a'), sideffect('b')],
        ...         provides=[sideffect('sum')])(addcolumns)
        ... )
        >>> graph
        NetworkOperation(name='mygraph', needs=[optional('df'), optional('sideffect(a)'), optional('sideffect(b)')], provides=['sideffect(sum)'])

        >>> # The graph works with and without 'c' provided as input.
        >>> df = pd.DataFrame({'a': [5], 'b': [2]})         # doctest: +SKIP
        >>> graph({'df': df})['sum'] == 11                  # doctest: +SKIP
        True

    Note that regular data in *needs* and *provides* do not match same-named *sideffects*.
    That is, in the following operation, the ``prices`` input is different from
    the ``sideffect(prices)`` output:

        >>> def upd_prices(sales_df, prices):
        ...     sales_df["Prices"] = prices

        >>> operation(fn=upd_prices,
        ...           name="upd_prices",
        ...           needs=["sales_df", "price"],
        ...           provides=[sideffect("price")])
        operation(name='upd_prices', needs=['sales_df', 'price'], provides=['sideffect(price)'], fn=upd_prices)

    .. note::
        An ``operation`` with *sideffects* outputs only, have functions that return
        no value at all (like the one above).  Such operation would still be called for
        their side-effects.

    .. tip::
        You may associate sideffects with other data to convey their relationships,
        simply by including their names in the string - in the end, it's just a string -
        but no enforcement will happen from *graphkit*.

        >>> sideffect("price[sales_df]")
        'sideffect(price[sales_df])'

    """

    __slots__ = ()  # avoid __dict__ on instances

    def __new__(cls, name):
        return super(sideffect, cls).__new__(cls, "sideffect(%s)" % name)
