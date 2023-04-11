"""
Suite of tools for eyetracking and neural net training

Created by AB 2021.10
"""
from .pylids import *

# downloading model weights
import os
import appdirs

user_config_dir = appdirs.user_config_dir()
if not os.path.exists(os.path.join(user_config_dir, 'pylids')):
    os.mkdir(os.path.join(user_config_dir, 'pylids'))