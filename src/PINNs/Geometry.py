import torch
from .Utility import *
from .Physics import NVS

def string_to_grad(input:str):
    """""
    example use
    input: 'p'_'x'
    output: ['p','x']
    """
    character_list = input.split('_')
        
    return character_list

class Bound():
    def __init__(self, range_x, func_x, is_inside, ref_axis='x', is_true_bound=True, func_n_x=None, func_n_y=None, range_n=None):
        self.range_x = range_x
        self.func_x = func_x
        self.is_inside = is_inside
        self.ref_axis = ref_axis
        self.is_true_bound = is_true_bound
    
        self.func_n_x = func_n_x #for circle
        self.func_n_y = func_n_y
        self.range_n = range_n
    def sampling_line(self, n_points, random=False):
        if self.func_n_x is None:
            if random:
                X = torch.empty(n_points).uniform_(self.range_x[0], self.range_x[1])
            else:
                X = torch.linspace(self.range_x[0], self.range_x[1], n_points)
            Y = self.func_x(X)
        else:
            if random:
                N = torch.empty(n_points).uniform_(self.range_n[0], self.range_n[1])
            else:
                N = torch.linspace(self.range_n[0], self.range_n[1], n_points)
            X = self.func_n_x(N)
            Y = self.func_n_y(N)

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
    def sampling_area(bound_list, n_points_square, range_x:list, range_y:list, random=False, return_info=False):
        if random:
            points = torch.empty(n_points_square, 2)
            points[:, 0].uniform_(range_x[0] + 1e-6, range_x[1] - 1e-6)  # x values
            points[:, 1].uniform_(range_y[0] + 1e-6, range_y[1] - 1e-6)  # y values
            X = points[:, 0]  # x-coordinates
            Y = points[:, 1]  # y-coordinates
        else:
            if isinstance(n_points_square, list):
                n_points_square_x = n_points_square[0]
                n_points_square_y = n_points_square[1]
            else:
                n_points_square_x = n_points_square_y = n_points_square
            X_range = torch.linspace(range_x[0]+1e-6, range_x[1]-1e-6, n_points_square_x)
            Y_range = torch.linspace(range_y[0]+1e-6, range_y[1]-1e-6, n_points_square_y)
            X, Y = torch.meshgrid(X_range, Y_range)
            X = X.flatten()
            Y = Y.flatten()

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


        if return_info:
            area_info = {
            "sampling_range_x" : range_x,
            "sampling_range_y" : range_y,
            "sampling_width"   : range_y[1] - range_y[0],
            "sampling_length"  : range_x[1] - range_x[0]
            }
            return X[~mask], Y[~mask], area_info
        else:
            return X[~mask], Y[~mask]
        
    @staticmethod
    def create_rectangle_bound(x_range:list , y_range:list, is_true_bound=True):
        if is_true_bound:
            i = lambda x : x
        else:
            i = lambda x : not x

        bound_list = [
            Bound(x_range, lambda x: y_range[0] * torch.ones_like(x), i(False), ref_axis='x', is_true_bound=is_true_bound), # Bottom wall
            Bound(x_range, lambda x: y_range[1] * torch.ones_like(x), i(True), ref_axis='x', is_true_bound=is_true_bound),  # Top wall
            Bound(y_range, lambda y: x_range[0] * torch.ones_like(y), i(False), ref_axis='y', is_true_bound=is_true_bound), # Inlet
            Bound(y_range, lambda y: x_range[1] * torch.ones_like(y), i(True), ref_axis='y', is_true_bound=is_true_bound) # Outlet
        ]

        return bound_list

    @staticmethod
    def create_circle_bound(x, y, r, is_true_bound = True):
        def func_up(X_tensor):
            return torch.sqrt(r**2 - (X_tensor-x)**2) + y
        def func_down(X_tensor):
            return -torch.sqrt(r**2 - (X_tensor-x)**2) + y
        
        def func_n_x_up(n):
            return x + r*torch.cos(n)
        def func_n_y_up(n):
            return y + r*torch.sin(n)
        def func_n_x_down(n):
            return x + r*torch.cos(n+torch.pi)
        def func_n_y_down(n):
            return y + r*torch.sin(n+torch.pi)

        if is_true_bound:
            i = lambda x : x
        else:
            i = lambda x : not x
        
        bound_list = [
        Bound([x-r,x+r], func_up, i(True), ref_axis='x', is_true_bound=is_true_bound, func_n_x=func_n_x_up, func_n_y=func_n_y_up, range_n = [0,torch.pi]),
        Bound([x-r,x+r], func_down, i(False), ref_axis='x', is_true_bound=is_true_bound, func_n_x=func_n_x_down, func_n_y=func_n_y_down, range_n = [0,torch.pi])
        ]
        return bound_list

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
        obj = cls()
        obj.bound = bound
        obj.condition_dict = condition_dict
        obj.range_t = range_t
        obj.bound_type = "BC"

        return obj
    
    @classmethod
    def define_initial_condition(cls, bound_list:list, sampling_range_x:list, sampling_range_y:list, condition_dict:dict, t=0.0):
        obj = cls()
        obj.bound_list = bound_list
        obj.sampling_range_x = sampling_range_x
        obj.sampling_range_y = sampling_range_y
        obj.condition_dict = condition_dict
        obj.t = t
        obj.bound_type = "IC"
        
        return obj
    
    @classmethod
    def define_pde_area(cls, bound_list:list, sampling_range_x:list, sampling_range_y:list, PDE_class:NVS, range_t=None):
        obj = cls()
        obj.bound_list = bound_list
        obj.range_t = range_t
        obj.sampling_range_x = sampling_range_x
        obj.sampling_range_y = sampling_range_y
        obj.PDE = PDE_class
        obj.bound_type = "PDE"

        return obj

    def sampling_collocation_points(self, n_points, random=False):
        self.is_sampled = True
        if self.bound_type == "BC":
            x,y = self.bound.sampling_line(n_points, random)
            
            if self.range_t is not None:
                t = PhysicsBound.sampling_time(self.range_t, self.number_points, random)
                t = t[:,None].requires_grad_()
            else:
                t = None

            x = x[:,None].requires_grad_()
            y = y[:,None].requires_grad_()

        else:    
            x, y, self.area_info = Bound.sampling_area(self.bound_list, n_points, self.sampling_range_x, self.sampling_range_y, random, True)
            if self.bound_type == 'IC':
                t = t*torch.ones_like(x)
                t = t[:,None].requires_grad_()
            else:
                t = None

            x = x[:,None].requires_grad_()
            y = y[:,None].requires_grad_()
        
        self.number_points = x.shape[0]

        self.inputs_tensor_dict = {'x':x,'y':y,'t':t}
        if self.bound_type == "IC" or self.bound_type == "BC":
            target_output_tensor_dict = {}
            for key in self.condition_dict:
                target_output_tensor_dict[key] = self.condition_dict[key] * torch.ones_like(x)
            self.target_output_tensor_dict = target_output_tensor_dict

        return x, y

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
    
    def calc_loss(self, model, loss_fn=None):

        if self.bound_type == "IC" or self.bound_type == "BC":
            pred_dict = self._calc_output(model)

            loss = 0
            for key in pred_dict:
                loss += loss_fn(pred_dict[key], self.target_output_tensor_dict[key])
            return loss/len(pred_dict)
        
        else:
            self.process_model(model)
            self.process_pde()
            return self.PDE.calc_loss()


#-----------------------------------------------------------------------------------------------process PDE related value
    def pde_define(self, pde_class):
        self.PDE = pde_class
    def process_model(self, model):
        self.model_inputs = self.inputs_tensor_dict
        self.model_outputs = model(self.inputs_tensor_dict)
        return self.model_outputs
    def process_pde(self):
        self.PDE.calc_residual(self.model_inputs | self.model_outputs)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch

    bound_list = []
    def func1(x):
        return 1*torch.ones_like(x)
    bound_list.append(Bound([0,2], func1, True))
    def func2(x):
        return 0*torch.ones_like(x)
    bound_list.append(Bound([0,2], func2, False))

    def func3(y):
        return 0*torch.ones_like(y)
    bound_list.append(Bound([0,1], func3, False, ref_axis='y'))
    def func4(y):
        return 2*torch.ones_like(y)
    bound_list.append(Bound([0,1], func4, True, ref_axis='y'))

    X, Y = Bound.sampling_area(bound_list, 200, [0,2], [0,1])
    plt.figure()
    plt.scatter(X,Y,s=1)
    for bound in bound_list:
        x,y = bound.sampling_line(200)
        plt.scatter(x,y,s=1, color='red')
    plt.xlim(-0.1,2.1)
    plt.ylim(-0.1,2.1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    physics_cond_list = [
        {'u': 100.0, 'v': 100.0},  # func1: top wall (y=1) -> No-slip
        {'u': 100.0, 'v': 100.0},  # func2: bottom wall (y=0) -> No-slip
        {'u': 100.0, 'v': 100.0},  # func3: inlet (x=0) -> Uniform inflow
        {'p': 101.0}             # func4: outlet (x=2) -> Zero pressure
    ]
    print(physics_cond_list[0])
    bound1 = PhysicsBound.define_boundary_condition(bound, physics_cond_list[0])
    bound2 = PhysicsBound.define_boundary_condition(bound, physics_cond_list[1])
    bound3 = PhysicsBound.define_boundary_condition(bound, physics_cond_list[2])
    bound4 = PhysicsBound.define_boundary_condition(bound, physics_cond_list[3])

    print(bound1.condition_dict)