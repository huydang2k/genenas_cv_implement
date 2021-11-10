#This module define cells in Nasgep architecture. See recurrrent_net.py
from copy import deepcopy
import math
import torch
import torch.nn as nn

from .module_tree import ModuleTree

from typing import List, Dict


class NasgepCellNet(nn.Module):
    def __init__(
        self,
        
        cells: List[ModuleTree],
        adfs: Dict,
        input_size: int,
        hidden_size: int,
        N: int = 3,
        batch_first: bool = False
    ):
        # Need to mimic Pytorch RNN as much as possible
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
       

        self.num_mains = len(cells)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        # self.layers = []
        for _ in range(num_layers):
            cell_list = nn.ModuleList()
            adfs_copy = deepcopy(adfs)
            for k, adf in adfs_copy.items():
                print('Init adf')
                print(adf)
                adf.init_tree(self.hidden_size)
                adfs_copy[k] = adf
            for cell in cells:
                new_cell = deepcopy(cell)
                print(new_cell)
                new_cell.init_tree(self.hidden_size)
                new_cell.assign_adfs(new_cell.root, adfs_copy)
                print('new cell')
                print(new_cell)
                cell_list.append(new_cell)
            self.layers.append(cell_list)
            # print(self.layers[-1]._modules)
        # print(self.layers)
        # self.hidden_size = self

        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            for cell in layer:
                std = 1.0 / math.sqrt(self.hidden_size)
                for weight in cell.parameters():
                    weight.data.uniform_(-std, std)

    # this code dumb, need more optimize
   
    def forward(self, x):
        print('foward NasgepCell')
        print(x)
        print('hidden state ', hidden_states)
        if self.batch_first:  # B x S x H
            # left_to_right_x = x
            # right_to_left_x = torch.flip(x, dims=[1])
            # x = torch.cat([left_to_right_x, right_to_left_x], dim=1)
            bs, w, h, c= x.size()
        else:  # S x B x H
            # left_to_right_x = x
            # right_to_left_x = torch.flip(x, dims=[0])
            # x = torch.cat([left_to_right_x, right_to_left_x], dim=0)
            seq_sz, bs, _ = x.size()

     
        
        for i, layer in enumerate(self.layers):
            if i == 0:
                seq_x = [self.fc(x[:, t, :].unsqueeze(0)) for t in range(seq_sz)]
                x = torch.cat(seq_x, dim=0)  # S x B x H
                if self.batch_first:
                    x = x.transpose(0, 1).contiguous()  # B x S x H
                x = torch.cat([x, x], dim=2)

            tmp_hidden_states = [states[i, :, :, :] for states in hidden_states]
            if self.bidirection:
                x, tmp_hidden_states = self.forward_bidirection(
                    layer, x, hidden_states=tmp_hidden_states
                )  # 2 x B x S
            else:
                x, tmp_hidden_states = self.forward_unidirection(
                    layer, x, hidden_states=tmp_hidden_states
                )  # 1 x B x S
            for main_id in range(self.num_mains):
                new_hidden_states[main_id].append(tmp_hidden_states[main_id])

        hidden_states = [torch.cat(states, dim=0) for states in new_hidden_states]

        return x, hidden_states