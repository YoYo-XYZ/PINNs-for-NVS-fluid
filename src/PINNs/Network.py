import torch
import torch.nn as nn
class PINN(nn.Module):
    """
    Physics-Informed Neural Network model.

    A feedforward neural network that takes {'x':, "y", t)
    and outputs the fluid velocity components u, v and pressure p.
    """
    def __init__(self, width, length, is_steady):

        super().__init__()
        self.is_steady = is_steady
        if self.is_steady == True:
            input_param = 2
        else:
            input_param = 3

        layers = []
        # Input layer
        layers.append(nn.Linear(input_param, width))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(length):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
            
        # Output layer
        layers.append(nn.Linear(width, 3))
        
        self.net = nn.Sequential(*layers)

        if is_steady:
            self.loss_history_dict = {'total_loss':[], 'bc_loss':[], 'pde_loss':[]}
        else:
            self.loss_history_dict = {'total_loss':[], 'bc_loss':[], 'ic_loss':[], 'pde_loss':[]}

    def forward(self, inputs_dict):
        """Forward pass through the network."""

        if self.is_steady:
            input_tensor = torch.cat([inputs_dict['x'], inputs_dict['y']], dim=1)
        else:
            input_tensor = torch.cat([inputs_dict['x'], inputs_dict['y'], inputs_dict['t']], dim=1)

        output = self.net(input_tensor)

        u = output[:, 0:1]
        v = output[:, 1:2]
        p = output[:, 2:3]
        
        return {'u':u, 'v':v, 'p':p}
    
    def show_updates(self):
        print(f"epoch {len(self.loss_history_dict['total_loss'])}, " + ", ".join(f"{key}: {value[-1]:.5f}" for key, value in self.loss_history_dict.items()))

#-----------------------------------------------------------------------
class NetworkTrainer():
    def __init__(self):
        self.optimizer_choice = {"Adam":None, "LBFGS":None}
    
    @staticmethod
    def record_loss(loss_hist_dict, loss_dict):
        for key in loss_dict:
            loss_hist_dict[key].append(loss_dict[key].item())
        return loss_hist_dict
    
    @staticmethod
    def train_adam(model, learning_rate, epochs, calc_loss, print_every=10):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss_dict = calc_loss()
            loss_dict['total_loss'].backward()
            optimizer.step()

            model.loss_history_dict = NetworkTrainer.record_loss(model.loss_history_dict, loss_dict)

            if epoch % print_every == 0:
                model.show_updates()
        return model

    @staticmethod
    def train_lbfgs(model, epochs, calc_loss, print_every=10):
        optimizer = torch.optim.LBFGS(model.parameters(), history_size=50, max_iter=20, line_search_fn="strong_wolfe")
        for epoch in range(epochs):
            def closure():
                optimizer.zero_grad()
                loss_dict = calc_loss()
                loss_dict['total_loss'].backward()
                closure.loss_dict = loss_dict
                return loss_dict['total_loss']
            optimizer.step(closure)
            loss_dict = closure.loss_dict
            model.loss_history_dict = NetworkTrainer.record_loss(model.loss_history_dict, loss_dict)
            if epoch % print_every == 0:
                model.show_updates()
        return model


