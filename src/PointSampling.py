import torch
from Utility import *
torch.manual_seed(43)

def string_to_grad(input:str):
    """""
    example use
    input: 'p'_'x'
    output: ['p','x']
    """
    character_list = input.split('_')
        
    return character_list

class Bound():
    def __init__(self, range_x, func_x, is_inside, ref_axis='x', is_true_bound=True):
        self.range_x = range_x
        self.func_x = func_x
        self.is_inside = is_inside
        self.ref_axis = ref_axis
        self.is_true_bound = is_true_bound
    
    def sampling_line(self, n_points, random=False):
        if random:
            X = torch.empty(n_points).uniform_(self.range_x[0], self.range_x[1])
            Y = self.func_x(X)
        else:
            X = torch.linspace(self.range_x[0], self.range_x[1], n_points)
            Y = self.func_x(X)
        
        if self.ref_axis == 'x':
            return X, Y
        else:
            return Y, X
    
    def mask_area(self, x, y):
        if self.ref_axis == 'y':
            x, y = y, x

        mask_x = (self.range_x[0] < x) & (x < self.range_x[1])
        if self.is_inside:
            mask_y = (y >= self.func_x(x))
        else:
            mask_y = (y <= self.func_x(x))

        return mask_x & mask_y

    @staticmethod
    def sampling_area(bound_list, n_points, range_x:list, range_y:list, random=False):
        if random:
            points = torch.empty(n_points**2, 2)
            points[:, 0].uniform_(range_x[0] + 1e-6, range_x[1] - 1e-6)  # x values
            points[:, 1].uniform_(range_y[0] + 1e-6, range_y[1] - 1e-6)  # y values
            X = points[:, 0]  # x-coordinates
            Y = points[:, 1]  # y-coordinates
        else:
            X_range = torch.linspace(range_x[0]+1e-6, range_x[1]-1e-6, n_points)
            Y_range = torch.linspace(range_y[0]+1e-6, range_y[1]-1e-6, n_points)
            X, Y = torch.meshgrid(X_range, Y_range)

        mask_list = []
        negative_mask_list = []
        for bound in bound_list:
            if bound.is_true_bound:
                mask_list.append(bound.mask_area(X,Y))
            else:
                negative_mask_list.append(bound.mask_area(X,Y))
        
        mask = torch.stack(mask_list, dim=0).any(dim=0)
        if negative_mask_list:
            negative_mask = torch.stack(negative_mask_list, dim=0).all(dim=0)
            mask = mask | negative_mask

        return X[~mask], Y[~mask]
        
import torch.nn as nn
class PhysicsBound():
    def __init__(self):
        self.is_sampled = False

    @staticmethod
    def sampling_time(range_t, n_points, random=False):
        range_t = range_t
        if random:
            t = torch.empty(n_points, 1).uniform_(range_t[0], range_t[1])
        else:
            t = torch.linspace(range_t[0], range_t[1], n_points).unsqueeze(1)
        return t
#----------------------------------------------------------------------------------------------- usual conditions
    @classmethod
    def define_boundary_condition(cls, bound:Bound, condition_dict: dict, range_t=None):
        cls.bound = bound
        cls.condition_dict = condition_dict
        cls.range_t = range_t
        cls.condition_type = "Boundary_conditions"

        return cls()
    
    @classmethod
    def define_initial_condition(cls, bound_list:list, sampling_range_x:list, sampling_range_y:list, condition_dict:dict, t=0.0):
        cls.bound_list = bound_list
        cls.sampling_range_x = sampling_range_x
        cls.sampling_range_y = sampling_range_y
        cls.condition_dict = condition_dict
        cls.t = t
        cls.condition_type = "Initial_conditions"
        
        return cls()

    def create_collocation_points(self, n_points, random):
        self.is_sampled = True
        if self.condition_type == "Boundary_conditions":
            x,y = self.bound.sampling_line(n_points, random)
            self.number_points = x.shape[0]
            
            if self.range_t is not None:
                t = PhysicsBound.sampling_time(self.range_t, self.number_points, random)

            x = x[:,None].requires_grad_()
            y = y[:,None].requires_grad_()
            t = t[:,None].requires_grad_()

        elif self.condition_type == "Initial_conditions":        
            x, y = Bound.sampling_area(self.bound_list, n_points, self.sampling_range_x, self.sampling_range_y, random)
            t = t*torch.ones_like(x)

            x = x[:,None].requires_grad_()
            y = y[:,None].requires_grad_()
            t = t[:,None].requires_grad_()

        else:
            print("Error: Condition error")

        self.inputs_tensor_dict = {'x':x,'y':y,'t':t}
        target_output_tensor_dict = {}
        for key in self.condition_dict:
            target_output_tensor_dict[key] = self.condition_dict[key] * torch.ones_like(x)
        self.target_output_tensor_dict = target_output_tensor_dict

#-----------------------------------------------------------------------------------------------process model's output and calculating loss
    def _calc_output(self, model):
        """"Post-process the model's output based on the target_dict keys."""
        prediction_dict = model(self.inputs_tensor_dict)
        pred_dict = {}

        for key in self.target_output_tensor_dict:
            if '_' in key:
                key_split = key.split('_')
                pred_dict[key] = calc_grad(prediction_dict[key_split[0]], self.target_output_tensor_dict[key_split[1]])
            else:
                pred_dict[key] = prediction_dict[key]
        return pred_dict
    
    def _loss_cal_each_condition(self, model, loss_fn):
        pred_dict = self._calc_output(model)

        loss = 0
        for key in pred_dict:
            loss += loss_fn(pred_dict[key], self.target_output_tensor_dict[key])
        return loss/len(pred_dict)
    
    def _define_pde(self, PDE):
        self.PDE = PDE
    
    def _get_pde_residual(self, model):
        return self.PDE.get_residual(model(self.inputs_tensor_dict))
    
    def _get_pde_loss(self):
        residual = self._get_pde_residual()
        return self.PDE.get_loss(residual)