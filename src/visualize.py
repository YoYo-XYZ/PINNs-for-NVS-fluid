import torch
import matplotlib.pyplot as plt
import numpy as np
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
            out_list.append(X.detach().numpy())
        return out_list

    @staticmethod
    def visualize_sol(model, x_range, y_range, n_points, t=None):
        X,Y = Visualization._create_grid(x_range, y_range, n_points)

        T=None
        if T is not None:
            T = t*torch.ones_like(X)

        U,V,P = Visualization._pred_from_model(model, X, Y, T, n_points)

        # Prepare data for plotting
        X_np, Y_np, U_np, V_np, P_np = Visualization._torch_to_numpy([X,Y,U,V,P])

        # Create plots
        plots_data = [
            (U_np, 'U velocity', 'viridis'),
            (V_np, 'V velocity', 'viridis'),
            (np.sqrt(U_np**2 + V_np**2), 'Velocity Magnitude', 'rainbow'),
            (P_np, 'Pressure', 'viridis'),
        ]
        fig, axes = plt.subplots(1, len(plots_data), figsize=(6*len(plots_data)/(y_range[1]-y_range[0])*(x_range[1]-x_range[0]), 5))

        for i, (data, title, cmap) in enumerate(plots_data):
            im = axes[i].contourf(X_np, Y_np, data, levels=80, cmap=cmap)
            axes[i].set_title(title)
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('y')
            plt.colorbar(im, ax=axes[i])
            
        plt.tight_layout()
        plt.show()

    def visualize_solution(model, time_ratio, n_points):
        """
        Generates and displays contour plots of the solution at a specific time ratio.
        """

        # Generate test data grid
        x_test = torch.linspace(0, 10, n_points)
        y_test = torch.linspace(0, 1, n_points)
        X, Y = torch.meshgrid(x_test, y_test, indexing='xy')
        T_eval = time_ratio * 1 * torch.ones_like(X)
        X.requires_grad_(True)
        Y.requires_grad_(True)
        T_eval.requires_grad_(True)

        # Predict solution using the model
        u, v, p = model(X.reshape(-1, 1), Y.reshape(-1, 1), T_eval.reshape(-1, 1))
        U = u.reshape(n_points, n_points)
        V = v.reshape(n_points, n_points)
        P = p.reshape(n_points, n_points)


        # Prepare data for plotting
        X_np, Y_np = X.detach().numpy(), Y.detach().numpy()
        U_np, V_np, P_np = U.detach().numpy(), V.detach().numpy(), P.detach().numpy()
        velocity_mag = np.sqrt(U_np**2 + V_np**2)

        # Create plots
        fig, axes = plt.subplots(1, 5, figsize=(30, 5))
        plots_data = [
            (U_np, 'U velocity', 'viridis'),
            (V_np, 'V velocity', 'viridis'),
            (velocity_mag, 'Velocity Magnitude', 'rainbow'),
            (P_np, 'Pressure', 'viridis'),
        ]

        for i, (data, title, cmap) in enumerate(plots_data):
            im = axes[i].contourf(X_np, Y_np, data, levels=80, cmap=cmap)
            axes[i].set_title(title)
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('y')
            plt.colorbar(im, ax=axes[i])
            
        plt.tight_layout()
        plt.show()

        return U_np, V_np, velocity_mag, P_np, total_residual_np

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