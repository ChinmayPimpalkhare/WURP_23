from create_dictionary import *
from inner_min import * 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, rc
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from pyswarms.single.global_best import GlobalBestPSO
from parameters import * 


optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=dim_theta, \
                          options=options, bounds=bounds_array)

def f(x):
    """Higher-level method to do forward_prop in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [-optimize_program_when_x_is_ellipsoid(x[i], **kwargs) for i in range(n_particles)]
    return np.array(j)