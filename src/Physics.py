import torch
from Utility import calc_grad

def string_to_grad(input:str):
    """""
    example use
    input: 'p'_'x'
    output: ['p','x']
    """
    character_list = input.split('_')
        
    return character_list

class PointSampling():
    def __init__(self):
        pass
#-----------------------------------------------------------------------------------------------sampling points generation
    @staticmethod
    def _sampling_range(range_A, num_points):
        if isinstance(range_A, (list, tuple)):
            A = torch.empty(num_points, 1).uniform_(range_A[0], range_A[1])
            A.requires_grad_()
        else:
            A = range_A * torch.ones(num_points, 1)
            A.requires_grad_()
        return A
    
    @staticmethod
    def _xyt_point_sampling(range_x, range_y, range_t, num_points, is_steady):
        x = PointSampling._sampling_range(range_x, num_points)
        y = PointSampling._sampling_range(range_y, num_points)
        if is_steady:
            t = None
        else:
            t = PointSampling._sampling_range(range_t, num_points)
        return x, y, t



class Conditions(PointSampling):
    def __init__(self, input:dict, target_output:dict):
        super().__init__()
        self.input = input
        self.target_output = target_output
#----------------------------------------------------------------------------------------------- usual conditions
    @classmethod
    def define_conditions(cls, range_x, range_y, range_t, num_points, is_steady, target_quantity_dict: dict):
        x,y,t = cls._xyt_point_sampling(range_x, range_y, range_t, num_points, is_steady)
        for key in target_quantity_dict:
            target_quantity_dict[key] = target_quantity_dict[key] * torch.ones(num_points, 1, requires_grad=True)
        return cls({'x':x,'y':y,'t':t}, target_quantity_dict)
    
    def nonslip_x(self, range_x, y, range_t=None, num_points=None):
        return self.define_conditions(range_x, y, range_t, num_points, target_quantity_dict = {'u':0.0, 'v':0.0, 'p_y': 0.0})
    
    def nonslip_y(self, x, range_y, range_t=None, num_points=None):
        return self.define_conditions(x, range_y, range_t, num_points, target_quantity_dict = {'u':0.0, 'v':0.0, 'p_x': 0.0})
    
    def outlet_y(self, x, range_y, range_t=None, num_points=None, pressure_value=0.0):
        return self.define_conditions(x, range_y, range_t, num_points, target_quantity_dict = {'u_x':0.0, 'v_x':0.0, 'p': pressure_value})
    
    def inlet_y_uniform(self, x, range_y, range_t=None, num_points=None, u_value=1):
        return self.define_conditions(x, range_y, range_t, num_points, target_quantity_dict = {'u':u_value, 'v':0.0, 'p_x': .0})
#-----------------------------------------------------------------------------------------------process model's output and condition loss
    def calc_output(self, model, inputs, target_dict:dict):
        prediction_dict = model(inputs)
        pred_dict = {}

        for key in target_dict:
            if '_' in key:
                key_split = key.split('_')
                pred_dict[key] = calc_grad(prediction_dict[key_split[0]], inputs[key_split[1]])
            else:
                pred_dict[key] = prediction_dict[key]
        return pred_dict
    
    def loss_cal(self, model, range_x, range_y, range_t=None, num_points=None, target_quantity_dict=None, loss_fn=None):
        inputs, target_dict = self.define_conditions(self, range_x, range_y, range_t, num_points, target_quantity_dict)
        pred_dict = self.calc_output(model, inputs, target_dict)
        
        loss = 0
        for key in pred_dict:
            loss += loss_fn(pred_dict[key], target_dict[key])
        loss = loss/len(pred_dict)
        return loss

class NVS(PointSampling):
    def __init__(self, is_steady = False):
        super().__init__()
        self.is_steady = is_steady

    def calc_nvs_residual(self, x, y, u, v, p, vu, rho, t=None):
        """
        Calculates the residuals of the incompressible Navier-Stokes equations.
        The residuals represent how well the network's predictions satisfy the PDEs.
        """
        # First-order derivatives

        if not self.is_steady:
            u_t = calc_grad(u, t)
            v_t = calc_grad(v, t)
        
        u_x = calc_grad(u, x)
        v_x = calc_grad(v, x)
        p_x = calc_grad(p, x)
        
        u_y = calc_grad(u, y)
        v_y = calc_grad(v, y)
        p_y = calc_grad(p, y)

        # Second-order derivatives
        u_xx = calc_grad(u_x, x)
        v_xx = calc_grad(v_x, x)
        u_yy = calc_grad(u_y, y)
        v_yy = calc_grad(v_y, y)
        
        # PDE residuals
        # Continuity equation (mass conservation)
        mass_residual = u_x + v_y
        
        if not self.is_steady:
            # X-momentum equation
            x_momentum_residual = (u_t + u * u_x + v * u_y -
                                vu * (u_xx + u_yy) +
                                p_x / rho)
                                
            # Y-momentum equation
            y_momentum_residual = (v_t + u * v_x + v * v_y -
                                vu * (v_xx + v_yy) +
                                p_y / rho)
                            
        else:
            # X-momentum equation
            x_momentum_residual = (u * u_x + v * u_y -
                                vu * (u_xx + u_yy) +
                                p_x / rho)
                                
            # Y-momentum equation
            y_momentum_residual = (u * v_x + v * v_y -
                                vu * (v_xx + v_yy) +
                                p_y / rho)

        return mass_residual, x_momentum_residual, y_momentum_residual


    def loss_cal(self, model, range_x, range_y, num_points, loss_fn, range_t=None):
        """Calculates the mean squared error of the PDE residuals."""
        x, y, t = self._xyt_point_sampling(range_x, range_y, range_t, num_points, self.is_steady)
        u_pred, v_pred, p_pred = model(x, y, t)
        
        mass_res, x_mom_res, y_mom_res = self.calc_nvs_residual(x, y, t, u_pred, v_pred, p_pred)
        
        pde_loss = (loss_fn(mass_res, torch.zeros_like(mass_res)) +
                    loss_fn(x_mom_res, torch.zeros_like(x_mom_res)) +
                    loss_fn(y_mom_res, torch.zeros_like(y_mom_res)))
                    
        return pde_loss
