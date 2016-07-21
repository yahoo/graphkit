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
    This modifier designates an input value in ``needs`` as optional.  In other
    words, the ``operation`` can still run if it does not receive this input,
    but it will use this input if available.  For example, this will make
    input ``c`` optional::

        operation(name='some_op', needs=['a', 'b', optional('c')]), ...)

    """
    pass
