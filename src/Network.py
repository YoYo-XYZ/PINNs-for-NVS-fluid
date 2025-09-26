import torch
import torch.nn as nn

class PINN(nn.Module):
    """
    Physics-Informed Neural Network model.

    A feedforward neural network that takes (x, y, t)
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
            layers.append(nn.Linear(width, length))
            layers.append(nn.Tanh())
            
        # Output layer
        layers.append(nn.Linear(width, 3))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x, y, t=None):
        """Forward pass through the network."""

        if self.is_steady:
            input_tensor = torch.cat([x, y], dim=1)
        else:
            input_tensor = torch.cat([x, y, t], dim=1)

        output = self.net(input_tensor)

        u = output[:, 0:1]
        v = output[:, 1:2]
        p = output[:, 2:3]
        
        return {'u':u, 'v':v, 'p':p}

#-----------------------------------------------------------------------

class NetworkUtils():
    def __init__(self, model):
        self.loss_history = {"epoch":[], "total_loss":[], "bc_loss":[], "ic_loss":[], "pde_loss":[]}
        self.optimizer_choice = {"Adam":None, "LBFGS":None}
        self.model = model

    def record_loss(self, loss_hist_dict, loss):
        for key in loss_hist_dict:
            loss_hist_dict[key].append(loss[key])

    def train_adam(self, learning_rate, epochs, calc_loss):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = calc_loss()
            loss.backward()
            optimizer.step()

            self.loss_history = self.record(self.loss_history, loss)

    def train_lbfgs(self, epochs, calc_loss):
        optimizer = torch.optim.LBFGS(self.model.parameters(), history_size=20, max_iter=100, line_search_fn="strong_wolfe")
        for epoch in range(epochs):
            def closure():
                optimizer.zero_grad()
                loss = calc_loss()
                loss.backward()
                optimizer.step()
                closure.loss = loss
                return loss

            self.loss_history = self.record(self.loss_history, closure)