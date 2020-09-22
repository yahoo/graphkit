#!/bin/bash

# clean, or invalid files in packages
rm -vrf ./build/* ./dist/* ./*.pyc ./*.tgz ./*.egg-info
python setup.py sdist bdist_wheel

