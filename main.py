import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
from scipy import constants
from LaplaceSolver import Simulation, plot
from Particle import particle

def main():
    L = 0.2338
    l = 0.0364
    d = 0.01
    Vd = 30
    Va = 1

    alpha = 0.5*(l)*(L+0.5*(l))/d

    sim = Simulation(x_edge_len=L+l, y_edge_len=L+l)
    m = constants.electron_mass
    q = constants.e
    electron = particle(-q, m, 0, sim.x_edge_len/2, 350000, 0)

    #sim.make_conds(Vd = Vd, Va = Va, l=l, d=d)
    sim.get_boundary_cond("electron_beam.png", 30)
    sim.potential = sim.compute_potential(sim.boundary_cond, sim.fixed_bool, sim.n_iter)
    path_x, path_y, Fy = electron.solve_path(sim, 1, 0.000000001)

    plt.figure()
    plot(sim.xv, sim.yv, sim.potential, path_x, path_y)
    plt.show()
    
if __name__ == '__main__':
    main()

