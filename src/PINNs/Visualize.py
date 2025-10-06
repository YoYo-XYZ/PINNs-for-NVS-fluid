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
    def histplot(Data, ax:plt, title=None, bins='fd'):
        ax.hist(Data, bins, density=True, alpha=0.7, color="steelblue", edgecolor="black")
        ax.set_title(title)

    @staticmethod
    def scatter_points(X, Y, ax, s=1):
        ax.scatter(X, Y, s=s)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return ax
    
class Visualization(Visualizer):
    def __init__(self, bound:PhysicsBound, pinns_model:PINN):
        super().__init__()
        self.bound = bound
        self.model = pinns_model
        self.data_dict = {}

    def sampling_plot_points(self, points_x, points_y):
        self.X,self.Y = self.bound.sampling_collocation_points([points_x, points_y], False)
        self.X_np = self.X.detach().numpy().flatten()
        self.Y_np = self.Y.detach().numpy().flatten()

        self.width = self.bound.area_info["sampling_width"]
        self.length = self.bound.area_info["sampling_length"]
        self.ratio = self.length/self.width

    def process_model(self):
        data_dict = {}

        # main outputs
        data_dict = data_dict | self.bound.process_model(self.model)
        self.bound.process_pde()
        data_dict = data_dict | self.bound.PDE.var
        data_dict["residual"] = self.bound.PDE.calc_residual_sum() # residual

        data_dict["velocity_magnitude"] = torch.sqrt(data_dict["u"]**2 + data_dict["v"]**2)
        self.data_dict = data_dict

        return list(self.data_dict.keys()), list(self.model.loss_history_dict.keys())

    def plotcolor_all(self):
        subplots_num = len(self.data_dict)
        fig, axes = plt.subplots(1,subplots_num, figsize=(6*subplots_num*self.ratio,6))

        for i, (key,data) in enumerate(self.data_dict.items()):
            self.colorplot(self.X_np,self.Y_np, data.detach().numpy().flatten(),axes[i],key,'viridis',s=100)
        plt.tight_layout()
        plt.show()

        return fig
    
    def plotcolor_select(self, key_cmap_dict):
        key_cmap_dict = Visualization._keycmap_dict_process(key_cmap_dict)
        num_plots = len(key_cmap_dict)
        fig, axes = plt.subplots(1,num_plots, figsize=(6*num_plots*self.ratio,6))

        for i, (key,data) in enumerate(key_cmap_dict.items()):
            self.colorplot(self.X_np,self.Y_np, self.data_dict[key].detach().numpy().flatten(),axes[i],key,data,s=100)
        plt.tight_layout()
        plt.show()
        return fig

    def plot_residual_distribution(self):
        fig, ax = plt.subplots()
        ax = plt.subplot()
        self.histplot(self.data_dict["residual"].detach().numpy().flatten(), ax, "abs PDE residual", bins = 100)
        plt.show()

        return fig

    def plot_loss_evolution(self, log_scale=False, linewidth = 0.1):
        fig, ax = plt.subplots()
        for key in self.model.loss_history_dict: 
            ax.plot(self.model.loss_history_dict[key], label = key, linewidth = linewidth)
            if log_scale:
                ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss per Iteration")
        ax.legend()  
        plt.show()

        return fig

    @staticmethod
    def _keycmap_dict_process(plot_dict):
        key_and_cmap_dict = {}
        for key in plot_dict:
            if plot_dict[key] is None:
                cmap = 'viridis'
            else:
                cmap = plot_dict[key]
            key_and_cmap_dict[key] = cmap
        return key_and_cmap_dict