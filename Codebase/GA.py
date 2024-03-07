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
import pygad

def fitness_func(ga_instance, solution, solution_idx):
    """Higher-level method to compute the fitness value of the chromosome
    Inputs
    ------
    The instance of the GA, the solution that is passed, the index of the solution
    Returns
    -------
    A scalar which is the fitness of the chromosome
    """
    fitness_value, _ = optimize_program_when_x_is_ellipsoid(solution, **kwargs)
    return fitness_value


fitness_function = fitness_func

def on_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])
