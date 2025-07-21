"""
Suite of tools for eyetracking and neural net training

Created by AB 2021.10
"""
from .pylids import *

# downloading model weights
import os
import shutil
import appdirs

user_config_dir = appdirs.user_config_dir()
pylids_config_dir = os.path.join(user_config_dir, 'pylids')
weights_dest = os.path.join(pylids_config_dir, 'weights_index.txt')

if not os.path.exists(pylids_config_dir):
    os.makedirs(pylids_config_dir)

if not os.path.exists(weights_dest):
    shutil.copy(os.path.join(os.path.dirname(__file__), 'weights_index.txt'), weights_dest)