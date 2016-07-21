#!/usr/bin/env python
# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

from setuptools import setup


setup(
     name='graphkit',
     version=1.0,
     description='Tools for modeling data and image transforms in a network dag',
     author='Huy Nguyen, Arel Cordero, Pierre Garrigues, Rob Hess, Tobi Baumgartner, Clayton Mellina',
     author_email='huyng@yahoo-inc.com',
     url='http://github.com/yahoo/graphkit',
     packages=['graphkit'],
     install_requires=[
          'networkx',
          'pydot'
     ],
     tests_require=[
          'numpy'
     ],
     zip_safe=False
)
