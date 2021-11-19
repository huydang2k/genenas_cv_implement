from copy import deepcopy

import torch.nn as nn

from evolution import GeneType
from .tree import Node, Tree
# from util import clone_module
from typing import Optional, Dict, List


class ModuleNode(nn.Module):
    def __str__(self):
        return str(self.value)
    def __init__(
        self,
        node,
        function_set,
        is_cv_problem = False,
        is_reduction  = False
    ):  
        # print(function_set)
        # print('Init Module node with value: ', node.value)
        super().__init__()
        self.node = node
        self.function_set = function_set
        self.is_cv_problem = is_cv_problem
        self.is_reduction = is_reduction
    def init_node_module(self, dim):
        if (
            self.node.node_type == GeneType.TERMINAL
            or self.node.node_type == GeneType.ADF_TERMINAL
            or self.node.node_type == GeneType.ADF
        ):
            return
        # elif self.node.node_type == GeneType.ADF:
        #     self.node_module.init_tree_module(self.node_module.root, dim)
        #     self.node_module = self.node_module.root
        else:
           
            # print(self.function_set)
           
            if 'conv' in self.node.value:
                if self.node.value == 'point_wise_conv':
                    called_function_name = self.node.value
                else:
                    called_function_name = self.node.value[:len(self.node.value)-4]
                # print(' Conv call: ' , called_function_name )
                self.add_module(
                "node_module", getattr(self.function_set, called_function_name)(dim,self.node.value)
                )
            else:
                self.add_module(
                    "node_module", getattr(self.function_set, self.node.value)(dim)
                )

    def init_child_list(self, child_list):
        self.add_module("child_list", nn.ModuleList(child_list))

    def assign_adf(self, adf_dict):
        # print('Assign adf')
        # print(adf_dict[self.node.value])
        self.add_module("node_module", adf_dict[self.node.value])

    def forward(self, input_dict):
        # print('pass in tree with type: ',self.node.node_type,' and value: ', self.node.value,)
        
        
        if (
            self.node.node_type == GeneType.TERMINAL
            or self.node.node_type == GeneType.ADF_TERMINAL
        ):  
            
            return input_dict[self.node.value]
        elif self.node.node_type == GeneType.ADF:
            
            for i, child in enumerate(self.child_list):
                copy_input_dict = {}
                for k, v in input_dict.items():
                    copy_input_dict[k] = v.detach().clone()
                input_dict[f"t{i + 1}"] = child(copy_input_dict)
            # print('input dict: ',input_dict)
            return self.node_module(input_dict)
        return self.node_module(*[child(input_dict) for child in self.child_list])
    def make_reduction(self, reduction_pos_list : List[bool]):
        if self.node.node_type == GeneType.ADF:
            tmp_node_module = deepcopy(self.node_module)
            self.node_module = tmp_node_module
            self.node_module.make_stride_along_tree(is_adf = True, reduction_pos_list = reduction_pos_list)
        else:
            self.node_module.make_reduction(reduction_pos_list)
#Tree to decode chromosome
class ModuleTree(nn.Module):
    def __init__(self, symbols: List, arity: List, gene_types: List, function_set):
        super().__init__()

        self.symbols = symbols
        self.arity = arity
        self.gene_types = gene_types
        self.function_set = function_set
        # self.root: Optional[ModuleNode] = None
        self.tree_structure = Tree(symbols, arity, gene_types)

    def init_tree(self, default_dim):

        root = self.init_tree_module(self.tree_structure.root, default_dim)
        
        #equavilent to self.root = root
        
        self.add_module("root", root)
        # self.init_tree_module_list(self.root)
    # def make_reduction_adf(self,reduction_pos_list : List[bool]):
    #     terminal_set = []
    #     for i in range(len(reduction_pos_list)):
    #         if reduction_pos_list[i]:
    #             terminal_set.append("t"+str(i + 1))
    #     root = self.root
    #     self.make_stride_dfs
    def make_stride_along_tree(self,is_adf = False,reduction_pos_list : List[bool] = []):
        terminal_set = []
        if is_adf:
            for i in range(len(reduction_pos_list)):
                if reduction_pos_list[i]:
                    terminal_set.append("t"+str(i + 1))
        else:
            for i in range(1,4):
                terminal_set.append("x"+str(i))
                terminal_set.append("t"+str(i))
        root = self.root
        self.make_stride_dfs(root,terminal_set)
        
    def make_stride_dfs(self, parent: ModuleNode,terminal_set):
        reduction_bool_list = []
        #each c is a ModuleNode
       
        for c in parent.child_list:
            if c.node.value in terminal_set:
                reduction_bool_list.append(True)
            else:
                reduction_bool_list.append(False)
                self.make_stride_dfs(c,terminal_set)
        if any(reduction_bool_list):
            parent.make_reduction(reduction_bool_list)
            
        # print('init tree module')
        # print('Value node ' , node.value)
        # print(index_main)
        # module_child_list = []
        # for child in node.child_list:
        #     child_node = self.init_tree_module(child, default_dim,index_main = index_main)
        #     module_child_list.append(child_node)
        # if 
        # module_node = ModuleNode(node, self.function_set)
        # module_node.init_node_module(default_dim,index_main)
        # module_node.init_child_list(module_child_list)
        # return module_node

    def init_tree_module(self, node: Node, default_dim: int):
        # Postorder 
        
        
        module_child_list = []
        for child in node.child_list:
            child_node = self.init_tree_module(child, default_dim)
            module_child_list.append(child_node)
        # if 
        module_node = ModuleNode(node, self.function_set)
        module_node.init_node_module(default_dim)
        module_node.init_child_list(module_child_list)
        return module_node
    #Only call this func when 
    #DFS to call ModuleNode.assign_adfs
    def assign_adfs(self, node: ModuleNode, adf_dict: Dict):
        for child in node.child_list:
            if child.node.node_type == GeneType.ADF:
                child.assign_adf(adf_dict)
                # print(self.chromosome)
                # child.module = adf_dict[child.value]
            self.assign_adfs(child, adf_dict)
        if node.node.node_type == GeneType.ADF:
            node.assign_adf(adf_dict)

    def forward(self, input_dict: Dict):
        return self.root(input_dict)
    def __str__(self):
        return str(self.symbols)