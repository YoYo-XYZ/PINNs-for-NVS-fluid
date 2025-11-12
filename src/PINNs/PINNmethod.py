from .Geometry import Area, Bound
import matplotlib.pyplot as plt
import torch
import sympy as sp
class ProblemDomain():
    def __init__(self, bound_list:list[Bound], area_list:list[Area]):
        self.bound_list = bound_list
        self.area_list = area_list
        self.sampling_option = None
    
    def _format_condition_dict(self, obj, obj_type="Bound"):
        """Helper function to format condition dictionary for display."""

        def func_to_latex(func_list):
            v = func_list
            return f"${str(sp.latex(v[1](sp.symbols(v[0]))))}$"

        if hasattr(obj, 'condition_dict'):
            conditions = ', '.join([f"{k}={(str(v) if isinstance(v,(float,int)) else func_to_latex(v))}" for k, v in obj.condition_dict.items()])
            return conditions
        elif hasattr(obj, 'PDE'):
            return f"PDE: {obj.PDE.__class__.__name__}"
        return ""
        
    def __str__(self):
        return f"""number of bound : {len(self.bound_list)}
        {[f'{i}: {len(bound.X)}' for i, bound in enumerate(self.bound_list)]}
        , number of area : {len(self.area_list)}
        {[f'{i}: {len(area.X)}' for i, area in enumerate(self.area_list)]}"""

    def sampling_uniform(self, bound_sampling_res:list, area_sampling_res:list, device='cpu'):
        self.sampling_option = 'uniform'
        for i, bound in enumerate(self.bound_list):
            bound.sampling_line(bound_sampling_res[i])
            bound.process_coordinates(device)
        for i, area in enumerate(self.area_list):
            area.sampling_area(area_sampling_res[i])
            area.process_coordinates(device)

    def sampling_random_r(self, bound_sampling_res:list, area_sampling_res:list, device='cpu'):
        self.sampling_option = 'random_r'
        for i, bound in enumerate(self.bound_list):
            bound.sampling_line(bound_sampling_res[i], random=True)
            bound.process_coordinates(device)
        for i, area in enumerate(self.area_list):
            area.sampling_area(area_sampling_res[i], random=True)
            area.process_coordinates(device)

    def sampling_RAR(self, bound_top_k_list:list, area_top_k_list:list, model, bound_candidates_num_list:list=None, area_candidates_num_list:list=None, device='cpu'):
        self.sampling_option = self.sampling_option + ' + RAR'
        for i, bound in enumerate(self.bound_list):
            if bound_candidates_num_list is None:
                bound.sampling_residual_based(bound_top_k_list[i], model)
            else:
                # Create a temporary copy by saving current state
                original_X = bound.X.clone() if hasattr(bound, 'X') else None
                original_Y = bound.Y.clone() if hasattr(bound, 'Y') else None
                
                # Sample new candidates
                bound.sampling_line(bound_candidates_num_list[i], random=True)
                bound.process_coordinates(device)
                X, Y = bound.sampling_residual_based(bound_top_k_list[i], model)
                
                # Restore and concatenate
                if original_X is not None:
                    bound.X = torch.cat([original_X, X])
                    bound.Y = torch.cat([original_Y, Y])
                else:
                    bound.X = X
                    bound.Y = Y
            bound.process_coordinates(device)
        for i, area in enumerate(self.area_list):
            if area_candidates_num_list is None:
                area.sampling_residual_based(area_top_k_list[i], model)
            else:
                # Create a temporary copy by saving current state
                original_X = area.X.clone() if hasattr(area, 'X') else None
                original_Y = area.Y.clone() if hasattr(area, 'Y') else None
                
                # Sample new candidates
                area.sampling_area(area_candidates_num_list[i], random=True)
                area.process_coordinates(device)
                X, Y = area.sampling_residual_based(area_top_k_list[i], model)
                
                # Restore and concatenate
                if original_X is not None:
                    area.X = torch.cat([original_X, X])
                    area.Y = torch.cat([original_Y, Y])
                else:
                    area.X = X
                    area.Y = Y
            area.process_coordinates(device)
#------------------------------------------------------------------------------------------------
    def show_coordinates(self):
        plt.figure(figsize=(20,20))
        
        for i, area in enumerate(self.area_list): #plot areas
            plt.scatter(area.X,area.Y,s=2, color='black', alpha=0.3)
            condition_str = self._format_condition_dict(area, "Area")
            label = f"Area {i}\n{condition_str}" if condition_str else f"Area {i}"
            plt.text(
                area.x_center,
                area.y_center,
                label,
                fontsize=16,
                color='navy',
                fontstyle='italic',
                fontweight='bold',
                family='serif',
                ha='center',
                va='center',
                bbox=dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1)  # simple white background
            )
        for i, bound in enumerate(self.bound_list): #plot bounds
            plt.scatter(bound.X,bound.Y,s=5, color='red', alpha=0.5)
            condition_str = self._format_condition_dict(bound, "Bound")
            label = f"Bound {i}\n{condition_str}" if condition_str else f"Bound {i}"
            plt.text(
                bound.x_center,
                bound.y_center,
                label,
                fontsize=16,
                color='darkgreen',
                fontstyle='italic',
                fontweight='bold',
                family='serif',   # elegant serif font
                ha='center',
                va='center',
                bbox=dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1)  # simple white background
            )
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def show_setup(self, bound_sampling_res:list=None, area_sampling_res:list=None):
        plt.figure(figsize=(20,20))
        
        if bound_sampling_res is None:
            bound_sampling_res = [int((bound.range_x[1] - bound.range_x[0])/0.008) for bound in self.bound_list]
        if area_sampling_res is None:
            area_sampling_res = [[int(area.length/0.008), int(area.width/0.008)] for area in self.area_list]

        for i, area in enumerate(self.area_list): #plot areas
            area.sampling_area(area_sampling_res[i])
            plt.scatter(area.X,area.Y,s=2, color='black', alpha=0.2, marker='s')
            condition_str = self._format_condition_dict(area, "Area")
            label = f"Area {i}\n{condition_str}" if condition_str else f"Area {i}"
            plt.text(
                area.x_center,
                area.y_center,
                label,
                fontsize=16,
                color='navy',
                fontstyle='italic',
                fontweight='bold',
                family='serif',
                ha='center',
                va='center',
                bbox=dict(facecolor='white', alpha=0.2, edgecolor='none', pad=1)  # simple white background
            )
        for i, bound in enumerate(self.bound_list): #plot bounds
            x, y = bound.sampling_line(bound_sampling_res[i])
            plt.scatter(x,y,s=2, color='red', alpha=0.5)
            condition_str = self._format_condition_dict(bound, "Bound")
            label = f"Bound {i}\n{condition_str}" if condition_str else f"Bound {i}"
            plt.text(
                bound.x_center,
                bound.y_center,
                label,
                fontsize=16,
                color='darkgreen',
                fontstyle='italic',
                fontweight='bold',
                family='serif',   # elegant serif font
                ha='center',
                va='center',
                bbox=dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1)  # simple white background
            )
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()