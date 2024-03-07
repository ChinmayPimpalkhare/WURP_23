import numpy as np

# Problem Parameters
ellipsoid_constants = np.array([0.2, 0.15])
ellipsoid_centre = np.array([0.05, -0.06])
max_noise = 3*1e-2
dim_v = 2
dim_m = 1
max_degree = 1
len_dict = dim_m*(max_degree + 1)**dim_v
eta_bound = 100
u_max_norm = 5
print_point = 1
EPSILON_HAMILTONIAN = 1e-2
MAX_VAL = 10e8

#Derived parameters
dim_theta = len_dict*(dim_v - 1)
max_bound_array = 2* np.pi*np.ones(dim_theta)
min_bound_array = np.zeros(dim_theta)
bounds_array = (min_bound_array, max_bound_array)

#Inner Minimization Parameters
method_innermin = 'COBYLA'
#PSO Parameters
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
n_particles = 5
PSO_iters = 10

#GA Parameters
num_genes = dim_theta
num_generations = 5
sol_per_pop = 10
num_parents_mating = sol_per_pop
GA_low_bound = 0
GA_high_bound = 2*np.pi

#Plotting Parameters
k_discrete = 100 #if dim_v = 2
#Define kwargs
kwargs={"ellipsoid_constants" : ellipsoid_constants,\
        "ellipsoid_centre": ellipsoid_centre, \
        "max_noise": max_noise,\
        "len_dict": len_dict,\
        "dim_v" : dim_v,\
        "dim_m": dim_m, \
        "max_degree": max_degree,\
        "u_max_norm": u_max_norm,\
        "eta_bound" : eta_bound,\
        "print_point" : print_point}
