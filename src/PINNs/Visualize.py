import torch
import matplotlib.pyplot as plt
import numpy as np
from .Network import *
from .Geometry import Area, Bound
from .PhysicsInformedAttach import PhysicsAttach

class Template():
    def __init__(self):
        pass
    @staticmethod
    def colorplot(X, Y, Data, ax:plt, title=None, cmap='viridis', s=1):
        im = ax.scatter(X, Y, s=s ,c=Data, cmap=cmap, marker = 's')
        ax.set_title(title, fontweight='medium', pad=10, fontsize=13)
        ax.set_xlabel('x', fontstyle='italic', labelpad=0)
        ax.set_ylabel('y', fontstyle='italic', labelpad=0)

        # # Create a dedicated axis for the colorbar for better fitting
        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="2%", pad=0.2)

        plt.colorbar(im, pad=0.03, shrink=1.2)

        return ax

    @staticmethod
    def lineplot(X, Data, ax:plt, xlabel, ylabel):
        # Main line style
        ax.plot(X, Data, linewidth=2.0, color="navy")

        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)

        ax.figure.tight_layout()

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
    
class Visualizer(Template):
    def __init__(self, pinns_model:PINN, geometry:Area|Bound):
        super().__init__()
        self.model = pinns_model.to('cpu')
        self.data_dict = {}
        self.geometry = geometry
        self.is_preprocessed = False

    def sampling_line(self, n_points:int):
        self.geometry.sampling_line(n_points)

    def sampling_area(self, n_points_square:list[int]):
        self.geometry.sampling_area(n_points_square)

    def preprocess(self):
        self.is_preprocessed = True

        self.X,self.Y = self.geometry.X, self.geometry.Y
        self.X_np = self.X.detach().numpy().flatten()
        self.Y_np = self.Y.detach().numpy().flatten()

        self.width = self.geometry.width
        self.length = self.geometry.length
        try:
            self.ratio = self.length/self.width
            if self.ratio<1 or self.ratio>100:
                self.ratio = 5
        except ZeroDivisionError:
            self.ratio = 5

        data_dict = {}
        self.geometry.process_coordinates(device = 'cpu')

        # Possible Outputs -> store in data dict
        data_dict = data_dict | self.geometry.process_model(self.model)
        if "u" in data_dict and "v" in data_dict:
            data_dict["velocity_magnitude"] = torch.sqrt(data_dict["u"]**2 + data_dict["v"]**2)
        if self.geometry.physics_type == 'PDE':
            data_dict = data_dict |self.geometry.PDE.var
        if self.geometry.physics_type is not None:
            data_dict[f"{self.geometry.physics_type} residual"] = torch.sqrt(self.geometry.calc_loss_raw(self.model)) # residual
        self.data_dict = data_dict

        print(f"available_data: {tuple(self.data_dict.keys())+tuple(self.model.loss_history_dict.keys())}")
    
    def plot_data_on_geometry(self, key_cmap_dict, s=10, orientation='vertical', range_x:list=None, range_y:list=None):
        key_cmap_dict = self._keycmap_dict_process(key_cmap_dict)
        num_plots = len(key_cmap_dict)
        if orientation == 'vertical':
            fig, axes = plt.subplots(num_plots, 1, figsize=(2.5*self.ratio, 2.5*num_plots))
        elif orientation == 'horizontal':
            fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots*self.ratio, 6))
        if num_plots == 1:
            axes = [axes]

        for i, (key,data) in enumerate(key_cmap_dict.items()):
            #print(self.X_np,self.Y_np, self.data_dict[key].detach().numpy().flatten())
            self.colorplot(self.X_np,self.Y_np, self.data_dict[key].detach().numpy().flatten(),axes[i],key,data,s=s)
            axes[i].set_aspect('equal', adjustable='box')
            if range_x is None or range_y is None:
                axes[i].set_xlim(self.geometry.range_x[0], self.geometry.range_x[1])
                axes[i].set_ylim(self.geometry.range_y[0], self.geometry.range_y[1])
            else:
                axes[i].set_xlim(range_x[0], range_x[1])
                axes[i].set_ylim(range_y[0], range_y[1])
        plt.tight_layout()
        plt.show()
        return fig

    def plot_data(self, key_list):
        num_plots = len(key_list)
        fig, axes = plt.subplots(num_plots, 1)
        if num_plots == 1:
            axes = [axes]
        for i, key in enumerate(key_list):
            coord = self.Y_np if self.geometry.ref_axis == 'y' else self.X_np
            self.lineplot(coord, self.data_dict[key].detach().numpy().flatten(), axes[i],self.geometry.ref_axis, key)

        plt.tight_layout()
        plt.show()
        return fig



    def plot_residual_distribution(self):
        fig, ax = plt.subplots()
        ax = plt.subplot()
        self.histplot(self.data_dict[f"{self.geometry.physics_type} residual"].detach().numpy().flatten(), ax, "abs PDE residual", bins = 100)
        plt.show()

        return fig
    
    def plot_loss_residual_evolution(self, log_scale=False, linewidth = 0.1):
        pass

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
    
