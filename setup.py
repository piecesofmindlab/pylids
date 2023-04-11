#!/usr/bin/env python
import os
import shutil
import appdirs
from setuptools import setup, find_packages

requirements = [
    'numpy', 'scipy'] # file_io, docdb_lite

setup(name="""pylids""",
          version='0.01',
          description="""Suite of tools for eyetracking""",
          maintainer='Arnab Biswas',
          license='Unclear',
          url='',
          packages=['pylids'],
          long_description=open('README.md').read())

user_config_dir = appdirs.user_config_dir()
if not os.path.exists(os.path.join(user_config_dir, 'pylids')):
    os.mkdir(os.path.join(user_config_dir, 'pylids'))
shutil.copy('pylids/weights_index.txt', os.path.join(user_config_dir, 'pylids','weights_index.txt'))