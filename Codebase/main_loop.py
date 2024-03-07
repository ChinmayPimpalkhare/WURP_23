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
from PSO import * 
from GA import *
from verification import * 
import time

if __name__ == "__main__": 
    start_time = time.time()
    # Example usage with 3-dimensional values and m=3
    # degree = 2
    # values = [symbols(f'x{i}') for i in range(1,3)]  # 3-dimensional
    # m = 2
    # evaluated_polynomials = evaluate_polynomials(degree, values, m)
    # vector_dict = create_vector_dict(evaluated_polynomials, m)

    # # Print the resulting dictionary of vectors
    # for key, vector in vector_dict.items():
    #     print(f'{key}: {vector}')

    # #PSO
    # cost, pos = optimizer.optimize(f, iters = iters)
    # plot_cost_history(cost_history=optimizer.cost_history)
    # position_history = optimizer.pos_history
    # print(position_history)
    # print(len(position_history))
    # print(position_history[0].shape)

    ##GA
    # Example usage for a 3-dimensional hypercube with 4 parts in each dimension

    ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type='rws',
                       init_range_low=GA_low_bound, 
                       init_range_high=GA_high_bound,
                       gene_space=None,
                       save_solutions=True,
                       mutation_probability=None, 
                       on_generation=on_gen,
                       suppress_warnings=True)

    ga_instance.run()
    ga_instance.plot_fitness()
    ga_instance.plot_genes()
    #ga_instance.plot_new_solution_rate()

    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    eta_psi_solution = obtain_optimal_feedback(solution)
    optimal_symbolic_feedback = symbolic_feedback(dim_m, dim_v, max_degree, eta_psi_solution)
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"The optimal feedback obtained is = {optimal_symbolic_feedback}")
    plot_vector_fields(dim_v, ellipsoid_constants, ellipsoid_centre, \
                       k_discrete, eta_psi_solution)
    end_time = time.time()
    runtime_total = end_time - start_time
    runtime_per_loop = runtime_total/num_generations
    runtime_per_organism = runtime_per_loop/sol_per_pop
    print(f"Runtime Statistics. \n Total = {runtime_total}. \n \
          Per Generation = {runtime_per_loop}. \n \
          Per Organism = {runtime_per_organism}")

