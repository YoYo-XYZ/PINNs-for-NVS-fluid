import torch

torch.manual_seed(43)

def string_to_grad(input:str):
    """""
    example use
    input: 'p'_'x'
    output: ['p','x']
    """
    character_list = input.split('_')
        
    return character_list

class Bound():
    def __init__(self, range_x, func_x, is_inside, ref_axis='x', is_true_bound=True):
        self.range_x = range_x
        self.func_x = func_x
        self.is_inside = is_inside
        self.ref_axis = ref_axis
        self.is_true_bound = is_true_bound
    
    def sampling_line_random(self, n_points):
        X = torch.linspace(self.x_range[0], self.x_range[1], n_points)
        Y = self.func_x(X)
        X.requires_grad_()
        Y.requires_grad_()
        
        if self.ref_axis == 'x':
            return X, Y
        else:
            return Y, X

    def sampling_line_uniform(self, n_points):
        X = torch.empty(n_points).uniform_(self.x_range[0], self.x_range[1])
        Y = self.func_x(X)
        X.requires_grad_()
        Y.requires_grad_()

        if self.ref_axis == 'x':
            return X, Y
        else:
            return Y, X
    
    def mask_area(self, x, y):
        if self.ref_axis == 'y':
            x, y = y, x

        mask_x = (self.range_x[0] < x) & (x < self.range_x[1])
        if self.is_inside:
            mask_y = (y >= self.func_x(x))
        else:
            mask_y = (y <= self.func_x(x))

        return mask_x & mask_y

    @staticmethod
    def sampling_area_uniform(bound_list, n_points, range_x:list, range_y:list):
        X_range = torch.linspace(range_x[0]+1e-6, range_x[1]-1e-6, n_points)
        Y_range = torch.linspace(range_y[0]+1e-6, range_y[1]-1e-6, n_points)
        X, Y = torch.meshgrid(X_range, Y_range)

        mask_list = []
        negative_mask_list = []
        for bound in bound_list:
            if bound.is_true_bound:
                mask_list.append(bound.mask_area(X,Y))
            else:
                negative_mask_list.append(bound.mask_area(X,Y))
        
        mask = torch.stack(mask_list, dim=0).any(dim=0)
        if negative_mask_list:
            negative_mask = torch.stack(negative_mask_list, dim=0).all(dim=0)
            mask = mask | negative_mask

        return X[~mask], Y[~mask]
            
    @staticmethod
    def sampling_area_random(bound_list, n_points, range_x:list, range_y:list):
        points = torch.empty(n_points**2, 2)
        points[:, 0].uniform_(range_x[0] + 1e-6, range_x[1] - 1e-6)  # x values
        points[:, 1].uniform_(range_y[0] + 1e-6, range_y[1] - 1e-6)  # y values
        X = points[:, 0]  # x-coordinates
        Y = points[:, 1]  # y-coordinates


        mask_list = []
        negative_mask_list = []
        for bound in bound_list:
            if bound.is_true_bound:
                mask_list.append(bound.mask_area(X,Y))
            else:
                negative_mask_list.append(bound.mask_area(X,Y))
        
        mask = torch.stack(mask_list, dim=0).any(dim=0)
        if negative_mask_list:
            negative_mask = torch.stack(negative_mask_list, dim=0).all(dim=0)
            mask = mask | negative_mask

        return X[~mask], Y[~mask]

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def func1(x):
        return 1
    bound1 = Bound([0,2], func1, True)
    def func2(x):
        return 0.5
    bound2 = Bound([0,0.5], func2, False)
    def func3(x):
        return 0
    bound3 = Bound([0,2], func3, False)

    def func4(y):
        return 0
    bound4 = Bound([0.5,1], func4, False, ref_axis='y')
    def func5(y):
        return 2
    bound5 = Bound([0,1], func5, True, ref_axis='y')
    def func6(y):
        return 1
    bound6 = Bound([0,0.5], func6, False, ref_axis='y')
    def func7(x):
        return torch.sqrt(0.2**2-(x-1)**2)+0.75
    bound7 = Bound([0.8,1.2], func7, False,'x',False)
    def func8(x):
        return -torch.sqrt(0.2**2-(x-1)**2)+0.75
    bound8 = Bound([0.8,1.2], func8, True,'x',False)

    X, Y = Bound.sampling_area_random([bound1, bound2, bound3, bound4, bound5, bound6, bound7, bound8], 200, [0,2], [0,1])
    print(X.shape)

    plt.figure()
    plt.scatter(X,Y,s=1)
    plt.xlim(-0.1,2.1)
    plt.ylim(-0.1,2.1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()




class PointSampling():
    def __init__(self):
        pass

#-----------------------------------------------------------------------------------------------sampling points generation
    @staticmethod
    def _sampling_range(range_A, num_points):
        if isinstance(range_A, (list, tuple)):
            A = torch.empty(num_points, 1).uniform_(range_A[0], range_A[1])
            A.requires_grad_()
        else:
            A = range_A * torch.ones(num_points, 1)
            A.requires_grad_()
        return A

    @staticmethod
    def circle(r,x,y,points):
        angle = torch.linspace(0,torch.pi,points)
        X = x + r*torch.cos(angle)
        Y = y + r*torch.sin(angle)
        X.requires_grad_()
        Y.requires_grad_()
        return X,Y

    @staticmethod
    def circle_mask(r,x,y, X, Y):
        outbound_mask1 = (X>torch.sqrt(r**2-y**2))
        outbound_mask2 = (Y>torch.sqrt(r**2-x**2))

        return outbound_mask1 | outbound_mask2
    
    @staticmethod
    def rectangle_mask(W,L,X,Y):
        mask = (Y>L )
    

    @staticmethod
    def _xyt_custom_point_sampling(x, y, t=None):
        pass

    @staticmethod
    def _xyt_point_sampling(range_x, range_y, range_t, num_points):
        x = PointSampling._sampling_range(range_x, num_points)
        y = PointSampling._sampling_range(range_y, num_points)
        if range_t is None:
            t = None
        else:
            t = PointSampling._sampling_range(range_t, num_points)
        return x, y, t