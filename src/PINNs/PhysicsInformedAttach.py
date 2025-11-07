from .Physics import PDE
from .Utility import calc_grad
import torch

class PhysicsAttach():
    def __init__(self):
        self.is_sampled = False
#----------------------------------------------------------------------------------------------- usual conditions
    def define_bc(self, condition_dict: dict, range_t=None):
        """Define boundary conditions (BC)."""
        self.condition_dict = condition_dict
        self.condition_num = len(condition_dict)
        self.range_t = range_t
        self.physics_type = "BC"
    
    def define_ic(self, condition_dict:dict, t=0.0):
        """Define initial conditions (IC)."""
        self.condition_dict = condition_dict
        self.condition_num = len(condition_dict)
        self.t = t
        self.physics_type = "IC"
    
    def define_pde(self, PDE_class:PDE, range_t=None):
        """Define the PDE to enforce."""
        self.range_t = range_t
        self.PDE = PDE_class
        self.physics_type = "PDE"
#----------------------------------------------------------------------------------------------- input to PINNs
    def sampling_time(self, n_points, random=False):
        """Define the PDE to enforce."""
        if self.range_t is None:
            self.t = None
        else:
            if random:
                self.t = torch.empty(n_points).uniform_(self.range_t[0], self.range_t[1])
            else:
                self.t = torch.linspace()

    def process_coordinates(self):
        """Prepare coordinates data to be feed to PINNs"""
        self.X_ = self.X[:,None].requires_grad_()
        self.Y_ = self.Y[:,None].requires_grad_()

        if self.range_t is not None:
            self.T_ = self.t[:,None].requires_grad_()
            self.inputs_tensor_dict = {'x':self.X_,'y':self.Y_,'t':self.T_}
        else:
            self.inputs_tensor_dict = {'x':self.X_,'y':self.Y_, 't':None}
            
        if self.physics_type == "IC" or self.physics_type == "BC":
            target_output_tensor_dict = {}

            for key in self.condition_dict: #loop over condition
                if isinstance(self.condition_dict[key],(float,int)): #if condition is constant
                    target_output_tensor_dict[key] = self.condition_dict[key] * torch.ones_like(self.X_)
                else: #if condition varies function
                    variable_key = self.condition_dict[key][0]
                    func = self.condition_dict[key][1]
                    target_output_tensor_dict[key] = func(self.inputs_tensor_dict[variable_key].detach().clone())
            self.target_output_tensor_dict = target_output_tensor_dict

        return self.inputs_tensor_dict
#----------------------------------------------------------------------------------------------- process model's output and calculating loss
    def calc_output(self, model):
        """"Post-process the model's output to get predict (based from target_output_dict.)"""
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
        """Calculate loss from PINNs output"""
        if self.physics_type == "IC" or self.physics_type == "BC": 
            pred_dict = self.calc_output(model)

            loss = 0
            for key in pred_dict:
                loss += loss_fn(pred_dict[key], self.target_output_tensor_dict[key])
            return loss/len(pred_dict)
        
        else:
            self.process_model(model)
            self.process_pde()
            return self.PDE.calc_loss()

#----------------------------------------------------------------------------------------------- process PDE related value
    def process_model(self, model):
        """feeds the inputs data to the model, returning model's output"""
        self.model_inputs = self.inputs_tensor_dict
        self.model_outputs = model(self.inputs_tensor_dict)
        return self.model_outputs
        
    def process_pde(self):
        self.PDE.calc_residual(inputs_dict=self.model_inputs | self.model_outputs)