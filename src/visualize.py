import torch
import matplotlib.pyplot as plt
import numpy as np
from Physics import NVS
class Visualization():
    def __init__(self, model):
        self.model = model

    @staticmethod
    def _create_grid(x_range, y_range, n_points):
        x_test = torch.linspace(x_range[0], x_range[1], n_points)
        y_test = torch.linspace(y_range[0], y_range[1], n_points)
        X, Y = torch.meshgrid(x_test, y_test, indexing='xy')
        X.requires_grad_(True)
        Y.requires_grad_(True)

        return X, Y
    
    @staticmethod
    def _pred_from_model(model, X, Y, T, n_points):
        pred = model({'x':X.reshape(-1, 1), 'y':Y.reshape(-1, 1), 't':T})
        U = pred["u"].reshape(n_points, n_points)
        V = pred["v"].reshape(n_points, n_points)
        P = pred["p"].reshape(n_points, n_points)

        return U, V, P
    
    @staticmethod
    def _torch_to_numpy(in_list):
        out_list = []
        for X in in_list:
            try:
                out_list.append(X.detach().numpy())
            except:
                out_list.append(X.numpy())
        return out_list

    @staticmethod
    def plot_data(plot_data, coordinate_data, ax):
        im = ax.contourf(coordinate_data["x"], coordinate_data["y"], plot_data["data"], levels=80, cmap=plot_data["cmap"])
        ax.set_title(plot_data["title"])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)

        return ax

    @staticmethod
    def visualize_sol(model, x_range, y_range, n_points, t=None):
        X,Y = Visualization._create_grid(x_range, y_range, n_points)

        T=None
        if T is not None:
            T = t*torch.ones_like(X)

        U,V,P = Visualization._pred_from_model(model, X, Y, T, n_points)

        # Prepare data for plotting
        X_np, Y_np, U_np, V_np, P_np = Visualization._torch_to_numpy([X,Y,U,V,P])
        PDE_residual = NVS.calc_nvs_residual_overall(X,Y,U,V,P)


        # Create plots
        plots_data = [
            {'data': U_np, 'title':'U velocity', 'cmap':'viridis'},
            {'data': V_np, 'title':'V velocity', 'cmap':'viridis'},
            {'data': np.sqrt(U_np**2 + V_np**2), 'title':'Velocity Magnitude', 'cmap':'rainbow'},
            {'data': P_np, 'title':'Pressure', 'cmap':'RdBu'},
            {'data': PDE_residual.detach().numpy(), 'title':'PDEresidual', 'cmap':'viridis'}
        ]
        fig, axes = plt.subplots(1, len(plots_data), figsize=(6*len(plots_data)/(y_range[1]-y_range[0])*(x_range[1]-x_range[0]), 5))

        for i, plot_data in enumerate(plots_data):
            Visualization.plot_data(plot_data, {"x": X_np, "y": Y_np}, axes[i])

        plt.tight_layout()
        plt.show()

        return fig


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