import torch
import torch.nn as nn
from typing import List
class AbstractFunctionset:
    specific_set = [] 
    function_set_dictionary = { 
        #name : arity
        "element_wise_sum": 2,
        "concat": 2,
        "point_wise_conv": 1,
        "depth_wise_conv_3x3": 1,
        "depth_wise_conv_5x5": 1,
        "depth_wise_conv_3x5": 1,
        "depth_wise_conv_5x3": 1,
        "depth_wise_conv_1x7": 1,
        "depth_wise_conv_7x1": 1,
        "separable_depth_wise_conv_3x3": 1,
        "separable_depth_wise_conv_5x5": 1,
        "separable_depth_wise_conv_3x5": 1,
        "separable_depth_wise_conv_5x3": 1,
        "separable_depth_wise_conv_1x7": 1,
        "separable_depth_wise_conv_7x1": 1
    }
    @classmethod
    def return_func_name(cls):
        specific_function_dict = []
        for name in getattr(globals()[cls.__name__],"specific_set"):
            specific_function_dict.append({"name":name,"arity":AbstractFunctionset.function_set_dictionary[name]})
        return specific_function_dict
    @classmethod
    def return_func_dict(cls):
        specific_function_dict = {}
        for name in getattr(globals()[cls.__name__],"specific_set"):
            if "separable_depth_wise_conv" in name:
                specific_function_dict[name] = AbstractFunctionset.separable_depth_wise_conv
                continue
            if "depth_wise_conv" in name:
                specific_function_dict[name] = AbstractFunctionset.depth_wise_conv
                continue
            specific_function_dict[name] = getattr(AbstractFunctionset,name)
        return specific_function_dict
   

    @staticmethod
    def element_wise_sum(dim):
        return AddModule()
    @staticmethod
    def concat(dim):
        return ConcatModule(dim)
    @staticmethod
    def point_wise_conv(dim,name,is_reduction = False):
        cin = cout = dim[0]
        return Point_Wise_Conv(cin,cout)
    @staticmethod
    def separable_depth_wise_conv(dim, name = 'separable_depth_wise_conv_3x3',is_reduction=False):
        cin = cout = dim[0]
        kernel_size = (int(name.split('_')[-1].split('x')[0]),int(name.split('_')[-1].split('x')[1]))
        stride = 1
        if is_reduction:
            # input()
            stride = 2
        padding = (int((kernel_size[0] - 1)/2), int((kernel_size[1] - 1)/2))
        return Separable_Depth_Wise__Conv(cin,cout, kernel_size,stride = stride, padding = padding)
    @staticmethod
    def depth_wise_conv(dim, name = 'depth_wise_conv_3x3',is_reduction=False):
        cin = cout = dim[0]
        kernel_size = (int(name.split('_')[-1].split('x')[0]),int(name.split('_')[-1].split('x')[1]))
        stride = 1
        if is_reduction:
            stride = 2
        padding = (int((kernel_size[0] - 1)/2), int((kernel_size[1] - 1)/2))
        return Depth_Wise__Conv(cin,cout, kernel_size,stride = stride, padding = padding)

class NLPFunctionSet:
    @staticmethod
    def return_func_name():
        return [
            {"name": "element_wise_sum", "arity": 2},
            {"name": "element_wise_product", "arity": 2},
            {"name": "concat", "arity": 2},
            {"name": "blending", "arity": 3},
            {"name": "linear", "arity": 1},
            {"name": "sigmoid", "arity": 1},
            {"name": "tanh", "arity": 1},
            {"name": "leaky_relu", "arity": 1},
            {"name": "layer_norm", "arity": 1},
        ]

    @staticmethod
    def return_func_dict():
        return {
            "element_wise_sum": NLPFunctionSet.element_wise_sum,
            "element_wise_product": NLPFunctionSet.element_wise_product,
            "concat": NLPFunctionSet.concat,
            "blending": NLPFunctionSet.blending,
            "linear": NLPFunctionSet.linear,
            "sigmoid": NLPFunctionSet.sigmoid,
            "tanh": NLPFunctionSet.tanh,
            "leaky_relu": NLPFunctionSet.leaky_relu,
            "layer_norm": NLPFunctionSet.layer_norm,
        }

    @staticmethod
    # def element_wise_sum(dim_left, dim_right):
    #    return AddModule(dim_left, dim_right)
    def element_wise_sum(dim):
        return AddModule()

    @staticmethod
    # def element_wise_product(dim_left, dim_right):
    #    return ProductModule(dim_left, dim_right)
    def element_wise_product(dim):
        return ProductModule()

    @staticmethod
    def concat(dim):
        return ConcatModule(dim)

    @staticmethod
    # def blending(dim1, dim2, dim3):
    #    return BlendingModule(dim1, dim2, dim3)
    def blending(dim):
        return BlendingModule()

    @staticmethod
    def linear(dim):
        return nn.Linear(in_features=dim, out_features=dim)

    @staticmethod
    def sigmoid(dim):
        return nn.Sigmoid()

    @staticmethod
    def tanh(dim):
        return nn.Tanh()

    @staticmethod
    def leaky_relu(dim):
        return nn.LeakyReLU()

    @staticmethod
    def layer_norm(dim):
        return nn.LayerNorm(dim)

class CV_Main_FunctionSet(AbstractFunctionset):
    specific_set = ['element_wise_sum','concat']

class CV_ADF_FunctionSet(AbstractFunctionset):
    specific_set = ['element_wise_sum','point_wise_conv','depth_wise_conv_3x3','depth_wise_conv_5x5','depth_wise_conv_3x5','depth_wise_conv_5x3','depth_wise_conv_1x7','depth_wise_conv_7x1','separable_depth_wise_conv_3x3','separable_depth_wise_conv_5x5','separable_depth_wise_conv_3x5','separable_depth_wise_conv_5x3','separable_depth_wise_conv_1x7','separable_depth_wise_conv_7x1']

class Point_Wise_Conv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.pwc = nn.Conv2d(cin, cout, kernel_size=1)

    """
    Parametter 'reduction_pos_list' just for convention. Just ignore because Conv has 1 input
    """
    def make_reduction(self,reduction_pos_list : List[bool]):
        del self.pwc
        self.pwc = nn.Conv2d(self.cin, self.cout, kernel_size=1,stride = 2)
    def forward(self, x):
        x = self.pwc(x)
        return x

class Depth_Wise__Conv(nn.Module):
    def __init__(self, cin, cout,kernel_size,stride = 1, padding = 0):
        super(Depth_Wise__Conv, self).__init__()
        self.cin = cin
        self.cout = cout
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.depthwise = nn.Conv2d(cin, cin, kernel_size=kernel_size, padding=padding, stride =stride, groups=cin)

    """
    Parametter 'reduction_pos_list' just for convention. Just ignore because Conv has 1 input
    """
    def make_reduction(self,reduction_pos_list : List[bool]):
        del self.depthwise
        self.depthwise = nn.Conv2d(self.cin, self.cin, kernel_size= self.kernel_size, padding=self.padding, stride =2, groups=self.cin)
    def forward(self, x):
        out = self.depthwise(x)

        # input()
        return out

class Separable_Depth_Wise__Conv(nn.Module):
    def __init__(self, cin, cout,kernel_size,stride = 1, padding = 0):
        self.cin = cin
        self.cout = cout
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        super(Separable_Depth_Wise__Conv, self).__init__()
        self.depthwise = nn.Conv2d(cin, cin, kernel_size=kernel_size, padding=padding, stride =stride,groups=cin)
        self.pointwise = nn.Conv2d(cin, cout, kernel_size=1)
    
    """
    Parametter 'reduction_pos_list' just for convention. Just ignore because Conv has 1 input
    """
    def make_reduction(self,reduction_pos_list : List[bool]):
        del self.depthwise
        self.depthwise = nn.Conv2d(self.cin, self.cin, kernel_size= self.kernel_size, padding=self.padding, stride =2, groups=self.cin)

    def forward(self, x):
        out = self.depthwise(x)
        
        
        # input()
        out = self.pointwise(out)
        return out

class ReshapeModule(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        if dim_in != dim_out:
            self.fc = nn.Linear(dim_in, dim_out)
        else:
            self.fc = nn.Identity()

    def forward(self, x):
        x = self.fc(x)
        return x


class AddModule(nn.Module):
    def __init__(self):
        super().__init__()
        # self.reshape = ReshapeModule(dim_left, dim_right)
        self.is_cv_problem = False    

        #if is_cv_problem
        self.reduction_pos_list = [False,False]
    def make_reduction(self,reduction_pos_list : List[bool]):
        self.is_cv_problem = True
        self.reduction_pos_list = reduction_pos_list
        self.avg_stride2 = nn.AvgPool2d(kernel_size = 2, stride = 2)
    def forward(self, a, b):
        # a, b = self.reshape(a, b)
        if self.is_cv_problem:
            if self.reduction_pos_list[0]:
                a = self.avg_stride2(a)
            if self.reduction_pos_list[1]:
                b = self.avg_stride2(b)  
        if a.shape != b.shape:
            print("Shape mismatch")
        x = torch.add(a, b)
        return x


class ProductModule(nn.Module):
    def __init__(self):
        super().__init__()
        # self.reshape = ReshapeModule(dim_left, dim_right)

    def forward(self, a, b):
        # a, b = self.reshape(a, b)
        x = torch.mul(a, b)
        return x




class ConcatModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.reshape = ReshapeModule(dim_in, dim_out)
        # print('create concat')
        # print('dim')
        # print(2 * dim)
        #if nlp problem (dim is just an interger)
        if isinstance(dim,int):
            self.reshape = ReshapeModule(2 * dim, dim)
            self.is_cv_problem = False 
        #if cv problem (dim is a tuple of [cin, w, h])
        else: 
            self.reshape = Point_Wise_Conv(dim[0] * 2, dim[0])
            self.is_cv_problem = True   

        #if is_cv_problem
        self.reduction_pos_list = [False,False]
    def make_reduction(self,reduction_pos_list : List[bool]):
        self.is_cv_problem = True
        self.reduction_pos_list = reduction_pos_list
        self.avg_stride2 = nn.AvgPool2d(kernel_size = 2, stride = 2)
    def forward(self, a, b):
        
        #cv problem
        if self.is_cv_problem:
            if self.reduction_pos_list[0]:
                a = self.avg_stride2(a)
            if self.reduction_pos_list[1]:
                b = self.avg_stride2(b)
            
            out = torch.cat([a, b], dim=1)
        #nlp problem
        else:
            out = torch.cat([a, b], axis=-1)
        out = self.reshape(out)
        return out


class BlendingModule(nn.Module):
    def __init__(self, dim_in1, dim_in2, dim_in3):
        super().__init__()
        self.product1 = ProductModule(dim_in1, dim_in3)
        self.product2 = ProductModule(dim_in2, dim_in3)
        left_dim = max(dim_in1, dim_in3)
        right_dim = max(dim_in2, dim_in3)
        self.add = AddModule(left_dim, right_dim)

    def forward(self, a, b, c):
        left = self.product1(a, c)
        right = self.product2(b, 1 - c)
        return self.add(left, right)


class BlendingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.product1 = ProductModule()
        self.product2 = ProductModule()
        self.add = AddModule()

    def forward(self, a, b, c):
        left = self.product1(a, c)
        neg_c = torch.neg(torch.sub(c, 1))
        right = self.product2(b, neg_c)
        return self.add(left, right)
