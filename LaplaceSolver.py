import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
import numba
from numba import jit
from scipy import constants

class Simulation:

    def __init__(self, x_edge_len=3, y_edge_len=3, res=300, n_iter=10000):
        self.n_iter = n_iter
        self.x_res = int(res)
        self.y_res = int(res)
        self.x_edge_len = x_edge_len
        self.y_edge_len = y_edge_len
        x_edge = np.linspace(0, x_edge_len, self.x_res)
        y_edge = np.linspace(0, y_edge_len, self.y_res)
        self.xv, self.yv = np.meshgrid(x_edge, y_edge)
        self.boundary_cond = np.zeros((self.x_res, self.y_res))
        self.potential = self.boundary_cond
        self.fixed = np.zeros((self.x_res, self.y_res))
        self.fixed_bool = np.zeros((self.x_res, self.y_res), dtype=bool)

    def get_boundary_cond(self, file, V):
        img = io.imread(f'../Laplace-Solver/images/{file}')
        self.fixed_bool = img[:,:,3]>200
        
        img = img[:,:,0:3]
        img = color.rgb2gray(img) 
        img = np.flip(img, axis=0)
        
        HV_bool = img<0.30
        LV_bool = img>0.80
        self.fixed_bool = HV_bool + LV_bool 

        self.fixed = img
        self.boundary_cond[self.fixed_bool] = V*((self.fixed[self.fixed_bool])-0.5)
        self.potential = self.boundary_cond
        
    def make_conds(self, Vd, Va, l, d):
        upper_y = int(self.y_res/2 + self.y_res*d/(self.y_edge_len*2))
        lower_y = int(self.y_res/2 - self.y_res*d/(self.y_edge_len*2))
        
        grad = np.linspace(-0.5, 0.5, int(l*self.x_res/self.x_edge_len))
        
        for i in range(0, int(l*self.x_res/self.x_edge_len)):
            self.fixed_bool[upper_y][i] = True
            self.boundary_cond[upper_y][i] = Vd/2 + grad[i]*Va
            self.fixed_bool[lower_y][i] = True
            self.boundary_cond[lower_y][i] = -Vd/2 + grad[i]*Va

    @staticmethod
    @numba.jit("f8[:,:](f8[:,:], b1[:,:], i8)", nopython=True, nogil=True)
    def compute_potential(boundary_cond, fixed_bool, n_iter):
        potential = boundary_cond
        length = len(potential[0])
        for n in range(n_iter):
            for i in range(1, length-1):
                for j in range(1, length-1):
                    if not(fixed_bool[i][j]):
                        potential[i][j] = 1/4 * (potential[i+1][j] + potential[i-1][j] + potential[i][j+1] + potential[i][j-1])
        return potential

    def get_index(self, x):
        index = self.x_res*((x/self.x_edge_len))
        return int(index)        

def plot(xv, yv, func, path_x=0, path_y=0):
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    clr_plot = ax.contourf(xv, yv, func, 50)
    plt.plot(path_x, path_y, 'r-')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    fig.colorbar(clr_plot, label='V [V]')