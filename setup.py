#!/usr/bin/env python
# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import os
import re
import io
from setuptools import setup

LONG_DESCRIPTION = """
GraphKit is a lightweight Python module for creating and running ordered graphs
of computations, where the nodes of the graph correspond to computational
operations, and the edges correspond to output --> input dependencies between
those operations.  Such graphs are useful in computer vision, machine learning,
and many other domains.
"""

# Grab the version using convention described by flask
# https://github.com/pallets/flask/blob/master/setup.py#L10
with io.open('graphkit/__init__.py', 'rt', encoding='utf8') as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

setup(
     name='graphkit',
     version=version,
     description='Lightweight computation graphs for Python',
     long_description=LONG_DESCRIPTION,
     author='Huy Nguyen, Arel Cordero, Pierre Garrigues, Rob Hess, Tobi Baumgartner, Clayton Mellina',
     author_email='huyng@yahoo-inc.com',
     url='http://github.com/yahoo/graphkit',
     packages=['graphkit'],
     install_requires=[
          "networkx; python_version >= '3.5'",
          "networkx == 2.2; python_version < '3.5'",
          "boltons"  # for IndexSet
     ],
     extras_require={
          'plot': ['pydot', 'matplotlib']
     },
     tests_require=['numpy'],
     license='Apache-2.0',
     keywords=['graph', 'computation graph', 'DAG', 'directed acyclical graph'],
     classifiers=[
          'Development Status :: 5 - Production/Stable',
          'License :: OSI Approved :: Apache Software License',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Topic :: Scientific/Engineering',
          'Topic :: Software Development'
    ],
    platforms='Windows,Linux,Solaris,Mac OS-X,Unix'
)
