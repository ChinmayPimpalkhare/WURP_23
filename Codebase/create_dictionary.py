from sympy import symbols, lambdify, Array, MutableDenseNDimArray
import numpy as np
from itertools import product
import re
import numpy as np 

def evaluate_polynomials(degree, values, m):
    # Create symbolic variables x1, x2, ..., xv
    variables = symbols('x:{}'.format(len(values)))

    # Generate all combinations of exponents up to the given degree
    exponents = product(range(degree + 1), repeat=len(values))

    # Create and evaluate polynomials
    evaluated_polynomials = {}
    for exp in exponents:
        for i in range(1, m + 1):
            vector_name = f'Vector {exp + (i,)}'  # Include i in the vector name
            # Create the polynomial expression
            polynomial_expr = 1  # Initialize with 1 for multiplication
            for var, e in zip(variables, exp):
                polynomial_expr *= var**e

            # Evaluate the polynomial for the given values
            numerical_value = lambdify(variables, polynomial_expr, 'numpy')(*values)

            # Store the result in the dictionary
            evaluated_polynomials[vector_name] = numerical_value

    return evaluated_polynomials

def extract_inside_parentheses(input_string):
    # Define a regular expression pattern to match content inside parentheses
    pattern = r'\((.*?)\)'

    # Use re.findall to find all matches
    matches = re.findall(pattern, input_string)

    # Return the matches (content inside parentheses)
    return matches

def create_vector_dict(evaluated_polynomials, m):
    vector_dict = {}
    for key, value in evaluated_polynomials.items():
        # Extract exponents from the key
        location = int(extract_inside_parentheses(key)[0].split(',')[-1]) - 1
        #vector = np.zeros(m). 
        #Commented, to be changed. 
        vector_mutable = MutableDenseNDimArray(np.zeros((m), dtype=float))
        vector = np.array(vector_mutable).astype(np.float64)
        vector[location] = value  # Set the i-th element of the vector
        vector_dict[key] = vector

    return vector_dict

def create_symbolic_dict(evaluated_polynomials, m):
    vector_dict = {}
    for key, value in evaluated_polynomials.items():
        # Extract exponents from the key
        location = int(extract_inside_parentheses(key)[0].split(',')[-1]) - 1
        #vector = np.zeros(m). 
        #Commented, to be changed. 
        vector_mutable = MutableDenseNDimArray(np.zeros((m), dtype=float))
        vector_mutable[location] = value  # Set the i-th element of the vector
        vector_dict[key] = vector_mutable
    return vector_dict

def generate_scalars(input_dict, eta_array):
    result_dict = {}
    d = len(eta_array)
    i = 0
    for key, value in input_dict.items():
        result_dict[key] = eta_array[i]
        i += 1
    return result_dict

def compute_feedback(vector_dict, coeff_dict):
    """
    Compute the sum of matrix-vector products across keys in two dictionaries.

    Parameters:
    - dict1: Dictionary with keys and numpy arrays (vectors) as values.
    - dict2: Dictionary with keys and numpy arrays (matrices) as values.

    Returns:
    - The sum of matrix-vector products.
    """
    result = sum(coeff_dict[key]*vector_dict[key] for key in vector_dict)
    return result


if __name__ == '__main__': 
    print(1)
