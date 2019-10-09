#!/usr/bin/env python
# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import io
import os
import re
from setuptools import setup


with open("README.md") as f:
    long_description = f.read()

# Grab the version using convention described by flask
# https://github.com/pallets/flask/blob/master/setup.py#L10
with io.open('graphkit/__init__.py', 'rt', encoding='utf8') as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

plot_reqs = [
    "matplotlib",   # to test plot
    "pydot",        # to test plot
]
test_reqs = plot_reqs + [
    "pytest",
    "pytest-cov",
    "pytest-sphinx",
]

setup(
     name='graphkit',
     version=version,
     description='Lightweight computation graphs for Python',
     long_description=long_description,
     author='Huy Nguyen, Arel Cordero, Pierre Garrigues, Rob Hess, '
     'Tobi Baumgartner, Clayton Mellina, ankostis@gmail.com',
     author_email='huyng@yahoo-inc.com',
     url='http://github.com/yahoo/graphkit',
     packages=['graphkit'],
     install_requires=[
          "networkx; python_version >= '3.5'",
          "networkx == 2.2; python_version < '3.5'",
          "boltons"  # for IndexSet
     ],
     extras_require={
          'plot': plot_reqs,
          'test': test_reqs,
     },
     tests_require=test_reqs,
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
