import torch
import torch.nn as nn
from Utility import calc_grad
from PointSampling import PointSampling
class Conditions(PointSampling):
    def __init__(self, inputs:dict, target_output:dict):
        super().__init__()
        self.inputs = inputs
        self.target_output = target_output
#----------------------------------------------------------------------------------------------- usual conditions
    @classmethod
    def define_conditions(cls, range_x, range_y, range_t, num_points, target_quantity_dict: dict):
        x,y,t = cls._xyt_point_sampling(range_x, range_y, range_t, num_points)
        for key in target_quantity_dict:
            target_quantity_dict[key] = target_quantity_dict[key] * torch.ones(num_points, 1, requires_grad=True)
        return cls({'x':x,'y':y,'t':t}, target_quantity_dict)
    ### commonly used conditions
    def nonslip_x(range_x, y, range_t=None, num_points=None):
        return Conditions.define_conditions(range_x, y, range_t, num_points, target_quantity_dict = {'u':0.0, 'v':0.0, 'p_y': 0.0})
    def nonslip_y(x, range_y, range_t=None, num_points=None):
        return Conditions.define_conditions(x, range_y, range_t, num_points, target_quantity_dict = {'u':0.0, 'v':0.0, 'p_x': 0.0})
    def outlet_y(x, range_y, range_t=None, num_points=None, pressure_value=0.0):
        return Conditions.define_conditions(x, range_y, range_t, num_points, target_quantity_dict = {'u_x':0.0, 'v_x':0.0, 'p': pressure_value})
    def inlet_y_uniform(x, range_y, range_t=None, num_points=None, u_value=1):
        return Conditions.define_conditions(x, range_y, range_t, num_points, target_quantity_dict = {'u':u_value, 'v':0.0, 'p_x': .0})
#-----------------------------------------------------------------------------------------------process model's output and calculating loss
    @staticmethod
    def _calc_output(model, inputs, target_dict:dict):
        """"Post-process the model's output based on the target_dict keys."""
        prediction_dict = model(inputs)
        pred_dict = {}

        for key in target_dict:
            if '_' in key:
                key_split = key.split('_')
                pred_dict[key] = calc_grad(prediction_dict[key_split[0]], inputs[key_split[1]])
            else:
                pred_dict[key] = prediction_dict[key]
        return pred_dict
    
    @staticmethod
    def _loss_cal_each_condition(condition, model, loss_fn):
        pred_dict = Conditions._calc_output(model, condition.inputs, condition.target_output)
        loss = 0
        for key in pred_dict:
            # print(pred_dict[key])
            # print(condition.target_output[key])
            loss += loss_fn(pred_dict[key], condition.target_output[key])
        loss = loss/len(pred_dict)
        return loss
    
    @staticmethod
    def loss_calc(condition_list, model, loss_fn=None):
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        total_loss = 0
        for cond in condition_list:
            total_loss += Conditions._loss_cal_each_condition(cond, model, loss_fn)
        return total_loss

class NVS(PointSampling):
    def __init__(self, is_steady = False):
        super().__init__()
        self.is_steady = is_steady

    @staticmethod
    def calc_nvs_residual(x, y, u, v, p, vu=0.001/1000, rho=1000, t=None):
        """
        Calculates the residuals of the incompressible Navier-Stokes equations.
        The residuals represent how well the network's predictions satisfy the PDEs.
        """
        # First-order derivatives

        if t is None:
            u_t = None
            v_t = None
        else:
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
          
        if t is None:
            # X-momentum equation
            x_momentum_residual = (u * u_x + v * u_y -
                                vu * (u_xx + u_yy) +
                                p_x / rho)
                                
            # Y-momentum equation
            y_momentum_residual = (u * v_x + v * v_y -
                                vu * (v_xx + v_yy) +
                                p_y / rho)

        else:
            # X-momentum equation
            x_momentum_residual = (u_t + u * u_x + v * u_y -
                                vu * (u_xx + u_yy) +
                                p_x / rho)
                                
            # Y-momentum equation
            y_momentum_residual = (v_t + u * v_x + v * v_y -
                                vu * (v_xx + v_yy) +
                                p_y / rho)    

        return mass_residual, x_momentum_residual, y_momentum_residual

    @staticmethod
    def calc_nvs_residual_overall(x, y, u, v, p, vu=0.001/1000, rho=1000, t=None):
        mass_residual, x_momentum_residual, y_momentum_residual = NVS.calc_nvs_residual(x, y, u, v, p, vu, rho, t)
        residual = torch.abs(mass_residual) + torch.abs(x_momentum_residual) + torch.abs(y_momentum_residual)
        return residual

    @staticmethod
    def loss_cal(model, range_x, range_y, num_points, range_t=None):
        """Calculates the mean squared error of the PDE residuals."""

        x, y, t = NVS._xyt_point_sampling(range_x, range_y, range_t, num_points)
        pred = model({'x':x, 'y':y, 't':t})
        
        mass_res, x_mom_res, y_mom_res = NVS.calc_nvs_residual(x, y, pred['u'], pred['v'], pred['p'])
        
        pde_loss = torch.mean(mass_res**2 + x_mom_res**2 + y_mom_res**2)

        return pde_loss
