import torch
import matplotlib.pyplot as plt
import numpy as np
from .Network import *
from .PointSampling import *
from .Network import *

class Visualizer():
    def __init__(self):
        pass
    @staticmethod
    def colorplot(X, Y, Data, ax:plt, title=None, cmap='viridis', s=1):
        im = ax.scatter(X, Y, s=s ,c=Data, cmap=cmap, marker = 's')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)
        return ax

    @staticmethod
    def lineplot(X, Data, ax:plt, title=None):
        ax.plot(X, Data)
        ax.grid(True)
        ax.set_title(title)

        return ax

    @staticmethod
    def histplot(Data, ax:plt, title=None, bins=30):
        ax.hist(Data, bins)
        ax.title(title)

    @staticmethod
    def scatter_points(X, Y, ax, s=1):
        ax.scatter(X, Y, s=s)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return ax
    
class Visualization(Visualizer):
    def __init__(self, bound:PhysicsBound, pinns_model:PINN):
        super.__init__()
        self.bound = bound
        self.model = pinns_model

    def sampling_plot_points(self, points_x, points_y):
        self.bound.sampling_collocation_points(points_x, points_y, False)

    def colorplot_model_outputs(self):
        self.bound.process_model(self.model)
        X, Y, t = self.bound.model_inputs
        X_np = X.detach().numpy().flatten()
        Y_np = Y.detach().numpy().flatten()

        plot_data_dict = self.bound.model_outputs

        fig, axes = plt.subplots(1,1+len(plot_data_dict), figsize=(6*5,6))
        for key, data in enumerate(plot_data_dict):
            self.colorplot(X_np,Y_np, data.detach().numpy().flatten(),axes[key],key,'viridis',s=50)
        plt.tight_layout()
        plt.show()

    def colorplot_model_full(self):
        self.bound.process_model(self.model)
        X, Y, t = self.bound.model_inputs
        X_np = X.detach().numpy().flatten()
        Y_np = Y.detach().numpy().flatten()

        plot_data_dict = self.bound.model_outputs
        plot_data_dict["velocity_magnitude"] = torch.sqrt(plot_data_dict['u']**2 + plot_data_dict['v']**2)
        plot_data_dict["pde_residual"] = self.bound._get_pde_residual_sum()

        fig, axes = plt.subplots(1,1+len(plot_data_dict), figsize=(6*5,6))
        for key, data in enumerate(plot_data_dict):
            self.colorplot(X_np,Y_np, data.detach().numpy().flatten(),axes[key],key,'viridis',s=50)
        plt.tight_layout()
        plt.show()

    def residual_distribution(self):
        residuals = self.bound._get_pde_residual_sum()
        ax = plt.subplot()
        self.histplot(residuals.detach().numpy(), ax, "residual", bins = 100)
        plt.show()