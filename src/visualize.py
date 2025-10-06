import torch
import matplotlib.pyplot as plt
import numpy as np
from Network import *
from PointSampling import *

class Visualizer():
    def __init__(self):

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
    def __init__(self, bound:PhysicsBound, pinns_model):
        super.__init__()
        self.bound = bound
        self.model = pinns_model

    def sampling_plot_points(self, points_x, points_y):
        self.bound.sampling_collocation_points(points_x, points_y, False)

    def colorplot_inputs_outputs(self):
        self.bound.process_model(self.model)
        X, Y, t = self.bound.model_inputs
        X_np = X.detach().numpy().flatten()
        Y_np = Y.detach().numpy().flatten()

        fig, axes = plt.subplots(1,1+len(self.bound.model_outputs), figsize=(6*5,6))
        for key, data in enumerate(self.bound.model_outputs):
            self.colorplot(X_np,Y_np, data.detach().numpy().flatten(),axes[key],key,'viridis',s=50)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_bound(bound_list):
        X, Y = Bound.sampling_area(bound_list)
        Visualization.scatter_points(X, Y)

    def define_area(self, area_bound:PhysicsBound, sampling_points_x, sampling_points_y):
        self.area_bound = area_bound
        self.X, self.Y = self.area_bound.sampling_collocation_points([sampling_points_x,sampling_points_y], random=False)

    def define_line()

    def define_pde(self, pde_class):
        pde_class









def create_animated_solution(model, config):
    """
    Creates an animated visualization of the solution over time.
    """
    n_points = config["vis_n_points"]
    time_steps = config["anim_time_steps"]
    time_range = config["anim_time_range"]
    L, W, T = config["channel_length"], config["channel_width"], config["total_time"]
    
    time_ratios = np.linspace(time_range[0], time_range[1], time_steps)
    
    x_test = torch.linspace(0, L, n_points)
    y_test = torch.linspace(0, W, n_points)
    X, Y = torch.meshgrid(x_test, y_test, indexing='xy')
    X_np, Y_np = X.detach().numpy(), Y.detach().numpy()
    
    fig, axes = plt.subplots(1, 5, figsize=(30, 5))
    fig.suptitle('Fluid Dynamics Solution Over Time', fontsize=16)

    # Initialize plots and colorbars to get their references
    # We plot dummy data first. The animate function will update this.
    
    plots = []
    cbars = []
    cmaps = ['viridis', 'viridis', 'rainbow', 'viridis', 'viridis']
    titles = ['U velocity', 'V velocity', 'Velocity Magnitude', 'Pressure', 'PDE Residual']
    dummy_data = visualize_solution(model, time_ratios[0], config)
    for i in range(5):
        im = axes[i].contourf(X_np, Y_np, dummy_data, levels=80, cmap=cmaps[i])
        plots.append(im)
        cbars.append(plt.colorbar(im, ax=axes[i]))
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')


    def animate(frame):
        time_ratio = time_ratios[frame]
        
        # Prepare tensors for gradient calculation
        X_grad = X.clone().requires_grad_(True)
        Y_grad = Y.clone().requires_grad_(True)
        T_eval = time_ratio * T * torch.ones_like(X_grad, requires_grad=True)
        
        # Get model predictions
        u, v, p = model(X_grad.reshape(-1, 1), Y_grad.reshape(-1, 1), T_eval.reshape(-1, 1))
        U, V, P = [tensor.reshape(n_points, n_points) for tensor in (u, v, p)]

        # Calculate residuals
        mass_res, x_mom_res, y_mom_res = calc_nvs_residual(X_grad, Y_grad, T_eval, U, V, P, config)
        total_residual = torch.abs(mass_res + x_mom_res + y_mom_res)

        # Prepare data for plotting
        plot_arrays = [
            U.detach().numpy(),
            V.detach().numpy(),
            np.sqrt(U.detach().numpy()**2 + V.detach().numpy()**2),
            P.detach().numpy(),
            total_residual.detach().numpy()
        ]
        
        # Update each plot
        for i in range(5):
            axes[i].clear()
            im = axes[i].contourf(X_np, Y_np, plot_arrays[i], levels=80, cmap=cmaps[i])
            axes[i].set_title(f'{titles[i]} (t/T = {time_ratio:.3f})')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('y')
            # Update colorbar by replacing the mappable
            cbars[i].mappable.set_array(plot_arrays[i])
            cbars[i].update_normal(im)
            
        return plots

    anim = animation.FuncAnimation(fig, animate, frames=time_steps, interval=200, blit=False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    
    if config["save_animation"]:
        filename = config["animation_filename"]
        print(f"Saving animation as {filename}...")
        anim.save(filename, writer='pillow', fps=5)
        print("Animation saved!")
    
    plt.show()
    return anim