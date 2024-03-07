from create_dictionary import *
import numpy as np
from parameters import * 
from scipy.optimize import minimize
import time
import gradient_free_optimizers as gradfree 

def forward_model_theta_to_y(theta_flat, len_dict, dim_v, ellipsoid_constants, ellipsoid_centre):
  theta_reshape = theta_flat.reshape((len_dict, (dim_v - 1)))
    # Inputs: 1. Theta_flat, a numpy array of length |D|*(v - 1)
    #         2. len_dict = size of the dictionary = |D|
    #         3. dim_v = dimension of the v-space
    #         4. ellipsoid_constants, an  array of length v, which defines the ellipsoid's axis constants
    #         5. ellipsoid_centre, an array of length v, which defines the ellipse's center
    # Hidden: Theta_reshape, obtained by reshaping theta_flat to (|D|, v - 1)
    # Output: y, an array of size (|D|, v), obtained by using the forward model to switch from polar to Cartesian coordinates
  #Computing y
  y = np.ones((len_dict, dim_v))
  for d in range(0, len_dict):
    y[d] *= ellipsoid_constants
    for i in range(0, dim_v):
      if i < (dim_v - 1):
        for j in range(0, i):
          y[d][i] *= np.sin(theta_reshape[d][j])
        y[d][i] *= np.cos(theta_reshape[d][i])
      if i == (dim_v - 1):
        for j in range(0, i):
          y[d][i] *= np.sin(theta_reshape[d][j])
    y[d] += ellipsoid_centre
  #Computing grad_y
  grad_y = np.ones((len_dict, dim_v))
  for d in range(0, len_dict):
    grad_y[d] *= 2*(y[d] - ellipsoid_centre)/(ellipsoid_constants**2)
    norm_grad = np.linalg.norm(grad_y[d])
    grad_y[d] = grad_y[d]/norm_grad
  return y, grad_y


def compute_x_dot_noisy(y_array, y_norm_grad, feedback, dim_v, dim_m, max_noise, k):
  #Problem dependent part
  #Moore Greitzer
  f_array = np.zeros(dim_v)
  f_array[0] = -y_array[k][1] - 1.5*y_array[k][0]**2 -0.5*y_array[k][0]**3
  f_array[1] = 0
  g_matrix = np.array([[0], [1]])
  g_array = np.dot(g_matrix, feedback)
  w  = max_noise * y_norm_grad[k]
  h_array = np.array([[0, 1], [0, 0]])
  w_array = np.dot(h_array, w)
  #Test Case-1
  '''
  f_array = np.zeros(dim_v)
  f_array[0] = 1*y_array[k][0]
  f_array[1] = 1*y_array[k][1]
  g_matrix = np.array([[1, 0], [0, 1]])
  g_array = np.dot(g_matrix, feedback)
  w  = max_noise * y_norm_grad[k]
  h_array = np.array([[0, 0], [0, 0]])
  w_array = np.dot(h_array, w)
  '''
  return f_array + g_array + w_array

def is_u_admissible(feedback, max_norm):
  phi = (np.linalg.norm(feedback) - max_norm)
  return phi

#@title Inner Minimization Assuming Ellipsoidal $\mathbb{X}$


def optimize_program_when_x_is_ellipsoid(theta_array,  \
                                         ellipsoid_constants, ellipsoid_centre,\
                                          max_noise, len_dict, dim_v, \
                                            dim_m, max_degree, u_max_norm, \
                                              eta_bound, print_point = 0):
    start_time_in = time.time()
    y, y_norm_grad = forward_model_theta_to_y(theta_array, len_dict, dim_v, ellipsoid_constants, ellipsoid_centre)
    dimension_d = len_dict

    def objective_function(x):
      sum = 0
      for i in range(0, len(x)):
        sum += np.abs(x[i]**2)
      # sum = 0
      # for k in range(len_dict): 
      #   evaluated_polynomials = evaluate_polynomials(max_degree, y[k], dim_m)
      #   psi_dict = create_vector_dict(evaluated_polynomials, dim_m)
      #   eta_psi  = generate_scalars(psi_dict, x)
      #   feedback = compute_feedback(eta_psi, psi_dict)
      #   #Problem Dependent
      #   x_dot_noisy = compute_x_dot_noisy(y, y_norm_grad, feedback, dim_v, dim_m, max_noise, k)
      #   sum += -np.dot(y_norm_grad[k], x_dot_noisy)/np.linalg.norm(x_dot_noisy)
      return sum

    def bounds_function(i):
        return (None, None)  # x[i] >= 0, where None means no upper bound

    # Create a list of bounds using the dynamically created functions
    bounds = [bounds_function(i) for i in range(0, len_dict)]

    # Define the constraint function based on the provided expression
    def constraint_function1(x, k):
      evaluated_polynomials = evaluate_polynomials(max_degree, y[k], dim_m)
      psi_dict = create_vector_dict(evaluated_polynomials, dim_m)
      eta_psi  = generate_scalars(psi_dict, x)
      feedback = compute_feedback(eta_psi, psi_dict)
      #Problem Dependent
      x_dot_noisy = compute_x_dot_noisy(y, y_norm_grad, feedback, dim_v, dim_m, max_noise, k)
      return np.dot(y_norm_grad[k], x_dot_noisy) + EPSILON_HAMILTONIAN

    def constraint_function2(x, k):
      evaluated_polynomials = evaluate_polynomials(max_degree, y[k], dim_m)
      psi_dict = create_vector_dict(evaluated_polynomials, dim_m)
      eta_psi  = generate_scalars(psi_dict, x)
      feedback = compute_feedback(eta_psi, psi_dict)
      return is_u_admissible(feedback, u_max_norm)

    # Define a list to store the constraint functions
    constraint_functions1 = [lambda x, k=k: -constraint_function1(x, k) for k in range(dimension_d)]
    constraint_functions2 = [lambda x, k=k: -constraint_function2(x, k) for k in range(dimension_d)]
    merged_constraint_functions = constraint_functions1 + constraint_functions2

    # Create a list of constraints using the dynamically created functions
    constraints = [{'type': 'ineq', 'fun': func} for func in merged_constraint_functions]

    # Initial guess
    initial_guess = [1] * dimension_d

    # Minimization using scipy.optimize
    result = minimize(objective_function, initial_guess, method=method_innermin, bounds=bounds, constraints=constraints, \
                      options={'disp': False})

    # Return the optimal solution and objective function value
    evaluated_polynomials = evaluate_polynomials(max_degree, y[0], dim_m)
    psi_dict_optimal = create_vector_dict(evaluated_polynomials, dim_m)
    eta_psi_optimal  = generate_scalars(psi_dict_optimal, result.x)
    feedback_optimal = compute_feedback(psi_dict_optimal, eta_psi_optimal)
    #print("Optimal_Feedback: " ,feedback_optimal)
    end_time = time.time()
    # print("Optimal eta_psi: ", eta_psi_optimal)
    # print("Runtime: ", end_time - start_time)
    # print("Optimization Success: ", result.success)
    # print("Status/Termination Cause: ", result.message)
    end_time_in = time.time()
    print(f"This loop ran for {end_time_in - start_time_in} seconds")
    if result.success == True:
      if print_point == 0:
        return result.fun
      elif print_point == 1:
        return result.fun, result.x
    elif result.success == False:
      if print_point == 0:
        return MAX_VAL
      elif print_point == 1:
        return MAX_VAL, result.x
      

def forward_model_single(theta, dim_v, ellipsoid_constants, ellipsoid_centre):
    # Inputs: 1. Theta_flat, a numpy array of length |D|*(v - 1)
    #         2. len_dict = size of the dictionary = |D|
    #         3. dim_v = dimension of the v-space
    #         4. ellipsoid_constants, an  array of length v, which defines the ellipsoid's axis constants
    #         5. ellipsoid_centre, an array of length v, which defines the ellipse's center
    # Hidden: Theta_reshape, obtained by reshaping theta_flat to (|D|, v - 1)
    # Output: y, an array of size (|D|, v), obtained by using the forward model to switch from polar to Cartesian coordinates
  #Computing y
  y = np.ones(dim_v)
  y *= ellipsoid_constants
  for i in range(0, dim_v):
    if i < (dim_v - 1):
      for j in range(0, i):
        y[i] *= np.sin(theta[j])
      y[i] *= np.cos(theta[i])
    if i == (dim_v - 1):
      for j in range(0, i):
        y[i] *= np.sin(theta[j])
  y += ellipsoid_centre
  #Computing grad_y
  grad_y = np.ones(dim_v)
  grad_y *= 2*(y - ellipsoid_centre)/(ellipsoid_constants**2)
  norm_grad = np.linalg.norm(grad_y)
  grad_y = grad_y/norm_grad
  return y, grad_y

def compute_x_dot_noisy_single(y, y_norm_grad, feedback, dim_v, dim_m, max_noise, set_f_0 = False):
  #Problem dependent part
  #Moore Greitzer
  f_array = np.zeros(dim_v)
  f_array[0] = -y[1] - 1.5*y[0]**2 -0.5*y[0]**3
  f_array[1] = 0
  g_matrix = np.array([[0], [1]])
  g_array = np.dot(g_matrix, feedback)
  w  = max_noise * y_norm_grad * np.random.rand()
  h_array = np.array([[0, 1], [0, 0]])
  w_array = np.dot(h_array, w)
  #Test Case-1
  '''
  f_array = np.zeros(dim_v)
  f_array[0] = 1*y_array[k][0]
  f_array[1] = 1*y_array[k][1]
  g_matrix = np.array([[1, 0], [0, 1]])
  g_array = np.dot(g_matrix, feedback)
  w  = max_noise * y_norm_grad[k]
  h_array = np.array([[0, 0], [0, 0]])
  w_array = np.dot(h_array, w)
  '''
  if set_f_0 == False: 
    return f_array + g_array + w_array
  elif set_f_0 == True: 
    return g_array