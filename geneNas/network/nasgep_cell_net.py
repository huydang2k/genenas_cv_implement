#This module define cells in Nasgep architecture. See recurrrent_net.py
from copy import deepcopy
import math
import torch
import torch.nn as nn

from .module_tree import ModuleTree

from typing import List, Dict


class NasgepCellNet(nn.Module):
    pass