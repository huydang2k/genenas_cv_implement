#This module define cells in Nasgep architecture. See recurrrent_net.py
from copy import deepcopy
import math
import torch
import torch.nn as nn

from .module_tree import ModuleTree

from typing import List, Dict,Tuple


class NasgepCellNet(nn.Module):
    def __init__(
        self,
        
        cells: List[ModuleTree],
        adfs: Dict,
        hidden_shape: Tuple[int,int,int],
        N: int = 3,

    ):
        # Need to mimic Pytorch RNN as much as possible
        super().__init__()
        self.hidden_shape = hidden_shape

        self.num_mains = len(cells)
        self.N = N
        self.layers = nn.ModuleList()
        # self.layers = []
    
        self.cell_list = nn.ModuleList()
        adfs_copy = deepcopy(adfs)
        for k, adf in adfs_copy.items():
            adf.init_tree(self.hidden_shape)
            adfs_copy[k] = adf
        for index_main, cell in enumerate(cells):
           
            new_cell = deepcopy(cell)
           
            is_cv_problem = True
            new_cell.init_tree(self.hidden_shape)
            new_cell.assign_adfs(new_cell.root, adfs_copy)
            if index_main == 1:
                new_cell.make_stride_along_tree()
            self.cell_list.append(new_cell)
        self.init_weights()

    def init_weights(self):
        for cell in self.cell_list:
            std = 1.0 / math.sqrt(self.hidden_shape[1])
            for weight in cell.parameters():
                weight.data.uniform_(-std, std)

    # this code dumb, need more optimize
    
   
    def forward(self, x):
        

        bs, c, w, h = x.size()
        # feature_map = x
        input_dict = {"x1": x,"x2": torch.clone(x),"x3": torch.clone(x) }

        for _ in range(2):
            for i in range(self.N):

                input_dict["x1"] = input_dict["x2"] = input_dict["x3"] = self.cell_list[0](input_dict)
                

            input_dict["x1"] = input_dict["x2"] = input_dict["x3"] = self.cell_list[1](input_dict)
            

        return input_dict["x1"]