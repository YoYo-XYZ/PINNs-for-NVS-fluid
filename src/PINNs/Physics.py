import torch
from .Utility import calc_grad

class NVS():
    def __init__(self, vu=0.001/1000, rho=1000.0):
        super().__init__()
        self.vu = vu
        self.rho = rho

    def calc_residual(self, inputs_dict):
        x = inputs_dict['x']
        y = inputs_dict['y']
        t = inputs_dict['t']
        u = inputs_dict['u']
        v = inputs_dict['v']
        p = inputs_dict['p']

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
                                self.vu * (u_xx + u_yy) +
                                p_x / self.rho)
                                
            # Y-momentum equation
            y_momentum_residual = (u * v_x + v * v_y -
                                self.vu * (v_xx + v_yy) +
                                p_y / self.rho)

        else:
            # X-momentum equation
            x_momentum_residual = (u_t + u * u_x + v * u_y -
                                self.vu * (u_xx + u_yy) +
                                p_x / self.rho)
                                
            # Y-momentum equation
            y_momentum_residual = (v_t + u * v_x + v * v_y -
                                self.vu * (v_xx + v_yy) +
                                p_y / self.rho)    

        self.residuals = (mass_residual, x_momentum_residual, y_momentum_residual)

        return self.residuals

    def calc_residual_sum(self):
        residuals_abs_sum = 0
        for residual in self.residuals:
            residuals_abs_sum += torch.abs(residual)
        return residuals_abs_sum

    def calc_loss(self):
        loss = 0
        for residual in self.residuals:
            loss += residual**2

        loss = torch.mean(loss)
        return loss
