#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name="""pylids""",
          version='0.2',
          description="""Suite of tools for eyetracking""",
          maintainer='Arnab Biswas',
          license='MIT',
          url='https://github.com/piecesofmindlab/pylids',
          packages=['pylids'],
          package_data={
              'pylids':[
                'weights_index.txt'
                  ],
              },
          long_description=open('README.md').read())