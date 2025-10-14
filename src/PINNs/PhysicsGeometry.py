from .Geometry import Bound, Area
from .Physics import NVS
from .Utility import calc_grad
class PhysicsGeometry():
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
    def define_initial_condition(cls, area:Area, condition_dict:dict, t=0.0):
        obj = cls()
        obj.area = area
        obj.condition_dict = condition_dict
        obj.t = t
        obj.bound_type = "IC"
        
        return obj
    
    @classmethod
    def define_pde_area(cls, area:Area, PDE_class:NVS, range_t=None):
        obj = cls()
        obj.area = area
        obj.range_t = range_t
        obj.PDE = PDE_class
        obj.bound_type = "PDE"

        return obj

    def sampling_collocation_points(self, n_points, random=False):
        self.is_sampled = True
        if self.bound_type == "BC":
            x,y = self.bound.sampling_line(n_points, random)
            
            if self.range_t is not None:
                t = PhysicsGeometry.sampling_time(self.range_t, self.number_points, random)
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
                if isinstance(self.condition_dict[key],(float,int)):
                    target_output_tensor_dict[key] = self.condition_dict[key] * torch.ones_like(x)
                else:
                    variable_key = self.condition_dict[key][0]
                    func = self.condition_dict[key][1]
                    target_output_tensor_dict[key] = func(self.inputs_tensor_dict[variable_key].detach().clone())
                    #print(target_output_tensor_dict[key])
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
    bound1 = PhysicsGeometry.define_boundary_condition(bound, physics_cond_list[0])
    bound2 = PhysicsGeometry.define_boundary_condition(bound, physics_cond_list[1])
    bound3 = PhysicsGeometry.define_boundary_condition(bound, physics_cond_list[2])
    bound4 = PhysicsGeometry.define_boundary_condition(bound, physics_cond_list[3])

    print(bound1.condition_dict)