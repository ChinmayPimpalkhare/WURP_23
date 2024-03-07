import numpy as np
import matplotlib.pyplot as plt
from parameters import * 

max_noise2 = 0

def ellipse_params(c1, c2, a1, a2, theta):
    x1 = c1 + a1 * np.cos(theta)
    x2 = c2 + a2 * np.sin(theta)
    return x1, x2

# Function to compute f1 and f2
def compute_f(c1, c2, a1, a2, theta):
    x1, x2 = ellipse_params(c1, c2, a1, a2, theta)
    dir1 = compute_normal(theta,a1,a2)[1]
    f1 = -x2 - 1.5*x1**2 - 0.5*x1**3 + max_noise2*np.random.random()*dir1
    f2 = 0# Define your function here based on x1 and x2
    return f1, f2

# Function to compute g1 and g2
def compute_g(c1, c2, a1, a2, theta):
    x1, x2 = ellipse_params(c1, c2, a1, a2, theta)
    g1 = 0
    #g2 =  -0.00331751964229724*x1*x2 + 4.7127329059459e-7*x1 - 427.583943014243*x2 - 23.4117086141303
    g2 = -0.0322593619544504*x1*x2 + 0.322249116321592*x1 - 0.263313081787096*x2 - 0.0140708503808712
    return g1, g2
# Function to compute the normal vector

def compute_normal(theta, a1, a2):
    normal = np.array([np.cos(theta)/a1, np.sin(theta)/a2])
    unit_normal = normal/np.abs(np.linalg.norm(normal))
    return unit_normal


# Function to compute angle between two vectors
c1, c2 = ellipsoid_centre[0], ellipsoid_centre[1]
a1, a2 = ellipsoid_constants[0], ellipsoid_constants[1]
l_bound = 1.3*np.pi
u_bound = 1.4*np.pi
n_samples_low = 10
n_samples_high = 100
theta_values_ellipse = np.linspace(l_bound,u_bound, n_samples_high)
theta_values = np.linspace(l_bound,u_bound, n_samples_low)
theta_values_high_res = np.linspace(l_bound, u_bound, n_samples_high)


# Plotting
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
scaling_factor_f = 1
scaling_factor_g = 1
scaling_factor_fg = 1


axs[0, 0].set_title('Drift Vector Field')
x1, x2 = ellipse_params(c1, c2, a1, a2, theta_values)
x1e, x2e = ellipse_params(c1, c2, a1, a2, theta_values_ellipse)
axs[0, 0].plot(x1e, x2e, label='Ellipse', linewidth  = 3)
f1, f2 = compute_f(c1, c2, a1, a2, theta_values)
quiver = axs[0, 0].quiver(x1, x2, f1, f2, color='r', \
                          scale=1/scaling_factor_f, label='Vector Field')
axs[0, 0].set_aspect('equal')  # Set aspect ratio to ensure proper ellipse representation
axs[0, 0].legend()

axs[0, 1].set_title('Control Vector Field')
axs[0, 1].plot(x1e, x2e, label='Ellipse', linewidth  = 3)
g1, g2 = compute_g(c1, c2, a1, a2, theta_values)
quiver = axs[0, 1].quiver(x1, x2, g1*scaling_factor_g , g2*scaling_factor_g, \
                          color='b', scale=1/scaling_factor_g, label='Additional Vector Field')
axs[0, 1].set_aspect('equal')
axs[0, 1].legend()

axs[1, 0].set_title('Resultant Vector Field')
axs[1, 0].plot(x1e, x2e, label='Ellipse', linewidth  = 3, color = 'r')
quiver = axs[1, 0].quiver(x1, x2, (f1 + g1)*scaling_factor_fg, \
                          (f2 + g2)*scaling_factor_fg\
                          , color='g', scale=1/scaling_factor_fg, label='Resultant Vector Field'\
                )
normals = compute_normal(theta_values, a1, a2)
quiver = axs[1, 0].quiver(x1, x2, normals[0]*scaling_factor_fg, (normals[1])*scaling_factor_fg\
                          , color='b', scale=1/scaling_factor_fg)
axs[1, 0].set_aspect('equal')
axs[1, 0].legend()

dot_products = np.zeros_like(theta_values_high_res)
i = 0
for theta in theta_values_high_res:
  normal = compute_normal(theta, a1, a2)
  f1, f2 = compute_f(c1, c2, a1, a2, theta)
  g1, g2 = compute_g(c1, c2, a1, a2, theta)
  dot_product = ((f1 + g1)*normal[0] + (f2 + g2)*normal[1])/np.sqrt((f1 + g1)**2 + (f2 + g2)**2) \
  if np.sqrt((f1 + g1)**2 + (f2 + g2)**2) > 0 else 0 
  dot_products[i] = dot_product
  i += 1
print(max(dot_products))


axs[1,1].set_title('Normalized Dot Product (Cosine()) as a Function of theta')
axs[1,1].plot(theta_values_high_res, dot_products, color='blue', label='Scatter Plot')
plt.axhline(0, color='red', linestyle='--', label='y=0')

plt.show()