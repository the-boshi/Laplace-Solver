import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
import numba
from numba import jit
from scipy import constants

class particle():
    def __init__(self, q, m, x, y, vx, vy):
        self.q = q
        self.m = m
        self.x0 = x
        self.y0 = y
        self.vx0 = vx
        self.vy0 = vy
    
    def solve_path(self, sim, t_end, dt):
        potential = sim.potential
        Ey, Ex = np.gradient(-potential)
        
        Fx = self.q*Ex
        Fy = self.q*Ey
        x = []
        y = []
        x.append(self.x0)
        y.append(self.y0)

        t = np.linspace(start=0, stop=t_end, num=int(t_end/dt))
        vx = self.vx0
        vy = self.vy0

        for i in range(0,len(t)):
            x_index = sim.get_index(x[i])
            y_index = sim.get_index(y[i])
            vx += dt*Fx[y_index][x_index]/self.m
            vy += dt*Fy[y_index][x_index]/self.m
            x_next = x[i]+vx*dt
            y_next = y[i]+vy*dt
            if 0<=x_next<=sim.x_edge_len and 0<=y_next<=sim.y_edge_len:
                x.append(x_next)
                y.append(y_next)
            else:
                break
            
        return x, y, Fy