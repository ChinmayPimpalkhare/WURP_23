from sympy import symbols, Matrix

# Define symbolic variables
x1, x2 = symbols('x1 x2')

# Define the vectors
vector1 = Matrix([1, x1])
vector2 = Matrix([2, x2])

# Define the coefficients
coeff1 = 5
coeff2 = 10

# Compute the linear combination
linear_combination = coeff1 * vector1 + coeff2 * vector2

# Print the result
print(f"Vector 1: {vector1}")
print(f"Vector 2: {vector2}")
print(f"Linear Combination: {linear_combination}")
