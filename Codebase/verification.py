from create_dictionary import *
from inner_min import * 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, rc
import pyswarms as ps
import matplotlib.pyplot as plt
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from pyswarms.single.global_best import GlobalBestPSO
from parameters import * 
from GA import * 

def obtain_optimal_feedback(solution):
    _, eta_psi_optimal = optimize_program_when_x_is_ellipsoid(solution, **kwargs)
    return eta_psi_optimal

def symbolic_feedback(dim_m, dim_v, max_degree, eta_psi_optimal):
    symbolic_x = [symbols(f'x{i}') for i in range(1, dim_v + 1)] 
    basis_pnms = evaluate_polynomials(max_degree, symbolic_x, dim_m)
    basis_vecs_symb =  create_symbolic_dict(basis_pnms, dim_m)
    coefficients = generate_scalars(basis_vecs_symb, eta_psi_optimal)
    optimal_symbolic_feedback = sum(coefficients[key]*basis_vecs_symb[key] for key in coefficients)
    return optimal_symbolic_feedback

def create_hypercube_grid(n, k):
    # Generate a grid for each dimension
    grid = np.linspace(0, 2*np.pi*(1 + 1/k), k, endpoint=False)

    # Create the hypercube grid using meshgrid
    meshgrid_args = [grid] * n
    hypercube_grid = np.meshgrid(*meshgrid_args, indexing='ij')

    # Combine the grid points into a list of n-dimensional points
    points = np.vstack([dimension.flatten() for dimension in hypercube_grid]).T

    return points

def plot_vector_fields(dim_v, ellipsoid_constants, ellipsoid_centre, \
                       k_discrete, eta_psi_optimal):
    thetas = create_hypercube_grid(dim_v - 1, k_discrete)
    theta_list = []
    y_list = []
    f_vecs = []
    g_vecs = []
    f_gu_vecs = []
    dot_products = []
    for i in range(0, thetas.shape[0]):
        y, y_grad = forward_model_single(thetas[i], dim_v, ellipsoid_constants, ellipsoid_centre)
        evaluated_polynomials = evaluate_polynomials(max_degree, y, dim_m)
        psi_dict = create_vector_dict(evaluated_polynomials, dim_m)
        eta_psi  = generate_scalars(psi_dict, eta_psi_optimal)
        feedback = compute_feedback(eta_psi, psi_dict)
        computed_f = compute_x_dot_noisy_single(y, y_grad , np.zeros(dim_m), dim_v, dim_m, max_noise, False)
        computed_g = compute_x_dot_noisy_single(y, y_grad , feedback, dim_v, dim_m, max_noise, True)
        computed_f_gu_w = compute_x_dot_noisy_single(y, y_grad , feedback, dim_v, dim_m, max_noise, False)
        theta_list.append(thetas[i])
        y_list.append(y)
        f_vecs.append(computed_f)
        g_vecs.append(computed_g)
        f_gu_vecs.append(computed_f_gu_w)
        dot_product = np.dot(computed_f_gu_w, y_grad)/(np.linalg.norm(computed_f_gu_w)*np.linalg.norm(y_grad))\
        if (np.linalg.norm(computed_f_gu_w)*np.linalg.norm(y_grad)) > 0 else 0
        dot_products.append(dot_product)
    # Plotting
    if dim_v == 2: 
        print("Proceeding to plot in 2 dimensions.")
        x1_values = [point[0] for point in y_list]
        x2_values = [point[1] for point in y_list]
        f1_values = [point[0] for point in f_vecs]
        f2_values = [point[1] for point in f_vecs]
        g1_values = [point[0] for point in g_vecs]
        g2_values = [point[1] for point in g_vecs]
        fg1_values = [point[0] for point in f_gu_vecs]
        fg2_values = [point[1] for point in f_gu_vecs]
        scaling_factor = 2

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        axs[0, 0].plot(x1_values, x2_values, color='red')
        axs[0, 0].quiver(x1_values, x2_values, f1_values, f2_values, scale=1/scaling_factor, color='blue')
        axs[0, 0].set_title('Plot 2')

        axs[0, 1].plot(x1_values, x2_values, color='red')
        axs[0, 1].quiver(x1_values, x2_values, g1_values, g2_values, scale=1/scaling_factor, color='green')
        axs[0, 1].set_title('Plot 3')
        
        axs[1, 0].plot(x1_values, x2_values, color='red')
        axs[1, 0].quiver(x1_values, x2_values, fg1_values, fg2_values, scale=1/scaling_factor, color='black')
        axs[1, 0].set_title('Plot 4')

        plt.tight_layout()
        axs[1,1].plot(theta_list, dot_products, color='black')
        plt.axhline(0, color='red', linestyle='--', label='y=0')
        axs[1,1].set_title('Plot 5')

        plt.show()
    return 0

    
     
    

