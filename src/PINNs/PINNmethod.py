from .Geometry import Area, Bound
import matplotlib.pyplot as plt

class ProblemDomain():
    def __init__(self, bound_list:list[Bound], area_list:list[Area]):
        self.bound_list = bound_list
        self.area_list = area_list
        self.N = 0
        self.sampling_option = None
        
    def __str__(self):
        return f"""number of bound : {len(self.bound_list)}
        , number of area : {len(self.area_list)}"""

    def sampling_uniform(self, bound_sampling_res:list, area_sampling_res:list):
        self.sampling_option = 'uniform'
        print('sampling')

        for i, bound in enumerate(self.bound_list):
            bound.sampling_line(bound_sampling_res[i])
            bound.process_coordinates()
        for i, area in enumerate(self.area_list):
            area.sampling_area(area_sampling_res)
            area.process_coordinates()

    def sampling_random_r(self, bound_sampling_res:list, area_sampling_res:list):
        self.sampling_option = 'random_r'
        print('random sampling')
        for i, bound in enumerate(self.bound_list):
            bound.sampling_line(bound_sampling_res[i], random=True)
            bound.process_coordinates()
        for i, area in enumerate(self.area_list):
            area.sampling_area(area_sampling_res[i], random=True)
            area.process_coordinates()
        self.N += 1

    def show_coordinates(self):
        plt.figure(figsize=(20,20))
        
        for i, area in enumerate(self.area_list): #plot areas
            plt.scatter(area.X,area.Y,s=2, color='black', alpha=0.3)
            plt.text(
                area.sampling_length/2,
                area.sampling_width/2,
                f"Area {i}",
                fontsize=15,
                color='navy',
                fontstyle='italic',
                fontweight='bold',
                family='serif',
                bbox=dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1)  # simple white background
            )
        for i, bound in enumerate(self.bound_list): #plot bounds
            plt.scatter(bound.X,bound.Y,s=5, color='red', alpha=0.5)
            plt.text(
                bound.X[len(bound.X)//2],
                bound.Y[len(bound.Y)//2],
                f"Bound {i}",
                fontsize=15,
                color='darkgreen',
                fontstyle='italic',
                fontweight='bold',
                family='serif',   # elegant serif font
                bbox=dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1)  # simple white background
            )
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def show_setup(self, bound_sampling_res:list=None, area_sampling_res:list=None):
        plt.figure(figsize=(20,20))
        
        if bound_sampling_res is None:
            bound_sampling_res = [int((bound.range_x[1] - bound.range_x[0])/0.008) for bound in self.bound_list]
        if area_sampling_res is None:
            area_sampling_res = [[int(area.sampling_length/0.008), int(area.sampling_width/0.008)] for area in self.area_list]

        for i, area in enumerate(self.area_list): #plot areas
            area.sampling_area(area_sampling_res[i])
            plt.scatter(area.X,area.Y,s=5, color='black', alpha=0.3, marker='s')
            plt.text(
                area.sampling_length/2,
                area.sampling_width/2,
                f"Area {i}",
                fontsize=16,
                color='navy',
                fontstyle='italic',
                fontweight='bold',
                family='serif',
                bbox=dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1)  # simple white background
            )
        for i, bound in enumerate(self.bound_list): #plot bounds
            x, y = bound.sampling_line(bound_sampling_res[i])
            plt.scatter(x,y,s=5, color='red', alpha=0.5)
            plt.text(
                x[len(x)//2],
                y[len(y)//2],
                f"Bound {i}",
                fontsize=16,
                color='darkgreen',
                fontstyle='italic',
                fontweight='bold',
                family='serif',   # elegant serif font
                bbox=dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1)  # simple white background
            )
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()