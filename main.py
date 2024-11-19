import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# read file and return the transposed matrix
def read_numbers_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    input_data = []
    
    for line in lines:
        values = line.strip().split()
        row = [float(value) for value in values]
        input_data.append(row)
    return np.array(input_data).T

# setting b vectors for iterations
def set_B0():
    return np.array([0.1, 10, 21])

# matrix of the mathematical problem
def model_mtrx():
    c1, c2, c3, c4, m1, m2, m3 = sp.symbols('c1 c2 c3 c4 m1 m2 m3')
    A = [[0, 1, 0, 0, 0, 0],
         [-(c2 + c1) / m1, 0, c2 / m1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [c2 / m2, 0, -(c2 + c3) / m2, 0, c3 / m2, 0],
         [0, 0, 0, 0, 0, 1],
         [0, 0, c3 / m3, 0, -(c4 + c3) / m3, 0]]
    return sp.Matrix(A)

# custom derivative function
def numerical_derivative(func, var, values, delta=1e-6):
    # create a copy of values to modify for numerical differentiation
    values_plus = values.copy()
    values_minus = values.copy()

    # perturb the variable by delta to approximate the derivative
    values_plus[var] += delta
    values_minus[var] -= delta

    diff = (func.subs(values_plus) - func.subs(values_minus)) / (2 * delta)
    
    return diff

# derivatives with respect to variable
def get_derivative(y_vec, b_vec, b_values):
    derivs = [] 
    
    for y_i in y_vec:
        # for each b_i (as an index)
        for b_i in b_vec:
            derivative = numerical_derivative(y_i, b_i, b_values)
            derivs.append(derivative)
    
    cols_n = len(b_vec)
    der_matr = [derivs[i:i + cols_n] for i in range(0, len(derivs), cols_n)]
    return np.array(der_matr)

# compute Runge-Kutta step for u matrix and track each iteration for plotting
def get_u_matr(a_matr, b_matr, u_matr, h):
    b_arrayed = np.array(b_matr.tolist())
    k1 = h * (np.dot(a_matr, u_matr) + b_arrayed)
    k2 = h * (np.dot(a_matr, u_matr + k1 / 2) + b_arrayed)
    k3 = h * (np.dot(a_matr, u_matr + k2 / 2) + b_arrayed)
    k4 = h * (np.dot(a_matr, u_matr + k3) + b_arrayed)
    return u_matr + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# compute Runge-Kutta step for y matrix and track each iteration for plotting
def get_y(a_matr, y_cur, h):
    k1 = h * np.dot(a_matr, y_cur)
    k2 = h * np.dot(a_matr, y_cur + k1 / 2)
    k3 = h * np.dot(a_matr, y_cur + k2 / 2)
    k4 = h * np.dot(a_matr, y_cur + k3)
    return y_cur + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def custom_hstack(arr1, arr2):
    if arr1.shape[0] != arr2.shape[0]:
        raise ValueError("arrays must have the same number of rows to be stacked horizontally.")
    return np.concatenate((arr1, arr2), axis=1)

#custom matrix inverse function using Gaussian elimination
def matrix_inv(matrix):
    n = matrix.shape[0]
    identity = np.eye(n)
    
    #augment the original matrix with the identity matrix
    augmented_matrix = custom_hstack(matrix, identity)
    
    #perform Gaussian elimination to convert the left part of the augmented matrix to identity
    for i in range(n):
        if augmented_matrix[i, i] == 0:
            raise ValueError("matrix is singular and cannot be inverted.")
        
        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i, i]
        
        for j in range(i + 1, n):
            factor = augmented_matrix[j, i]
            augmented_matrix[j] = augmented_matrix[j] - factor * augmented_matrix[i]
    
    #back substitution to eliminate the upper triangle
    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            factor = augmented_matrix[j, i]
            augmented_matrix[j] = augmented_matrix[j] - factor * augmented_matrix[i]
    
    return augmented_matrix[:, n:]

# approximation method with plotting
def approximate(y_matr, params, beta_symbols, beta_values, eps, h=0.2):
    a_matrix = model_mtrx().subs(params) 
    beta_vector = np.array([0.1, 10, 21])

    # track iteration and error for plotting
    errors = []
    iterations = []
    approximated_y = []

    iteration = 0
    while True:
        a_complete = np.array((a_matrix.subs(beta_values)).tolist())  
        u_matr = np.zeros((6, 3))
        quality_degree = 0
        integral_part_inverse = np.zeros((3, 3))
        integral_part_mult = np.zeros((1, 3))
        y_approximation = y_matr[0]

        for i in range(len(y_matr)):
            b_derivative_matr = get_derivative(a_matrix * sp.Matrix(y_approximation), beta_symbols, beta_values)

            #accumulate integrals for the update rule
            integral_part_inverse += np.dot(u_matr.T, u_matr).astype('float64')
            integral_part_mult += np.dot(u_matr.T, y_matr[i] - y_approximation).astype('float64')

            #calculate the quality degree (error measure)
            quality_degree += np.dot((y_matr[i] - y_approximation).T, y_matr[i] - y_approximation)

            #update u_matr and y_approximation
            u_matr = get_u_matr(a_complete, b_derivative_matr, u_matr, h)
            y_approximation = get_y(a_complete, y_approximation, h)
            approximated_y.append(y_approximation)

        #scale integrals by step size
        integral_part_inverse *= h
        integral_part_mult *= h
        quality_degree *= h

        #solve for the parameter updates using the custom matrix inverse
        integral_part_inverse_inv = matrix_inv(integral_part_inverse)
        delta_beta = np.dot(integral_part_inverse_inv, integral_part_mult.flatten())
        beta_vector += delta_beta

        #update the parameter values
        beta_values = {
            beta_symbols[0]: round(beta_vector[0], 4),
            beta_symbols[1]: round(beta_vector[1], 4),
            beta_symbols[2]: round(beta_vector[2], 4)
        }

        # store iteration count and error for plotting
        errors.append(quality_degree)
        iterations.append(iteration)
        
        # print rounded outputs for readability
        print(f"Iteration {iteration}:")
        print(f"  Current approximated values: {beta_values}")
        print(f"  Quality Degree (Delta): {round(quality_degree, 6)}")

        # stop if the quality accuracy reached
        if quality_degree < eps:
            # number of subplots
            num_comparisons = len(y_matr[0])
            rows, cols = 2, 3 #setting plot display grid

            # plot convergence of the error
            plt.figure(figsize=(15, 12)) 
            plt.subplot(rows + 1, cols, 1)
            plt.plot(iterations, errors, marker='o')
            plt.xlabel("Iterations")
            plt.ylabel("Error")
            plt.title("Convergence of Parameter Approximation")

            # plot comparison of original and approximated `y`
            time_steps = np.arange(250) 

            for idx in range(num_comparisons):
                plt.subplot(rows + 1, cols, idx + 4)  # offset by 3 for the error plot
                original = y_matr[:250, idx]
                approximated = np.array(approximated_y)[:250, idx]

                plt.plot(time_steps, original, label=f"original y{idx + 1}")
                plt.plot(time_steps, approximated, linestyle='--', label=f"approximated y{idx + 1}")
                plt.legend()
                plt.xlabel("time steps")
                plt.ylabel("y")
                plt.title(f"comparison of original and approximated y{idx + 1}")
            
            # adjust layout for better spacing
            plt.tight_layout(h_pad=2.0, w_pad=1.5)
            plt.show()
            return beta_values

        print(f"  Delta is greater than {eps} -> moving to the next iteration\n")
        iteration += 1

if __name__ == "__main__":
    input_data = read_numbers_from_file('y1.txt')
    c1, c2, c3, c4, m1, m2, m3 = sp.symbols('c1 c2 c3 c4 m1 m2 m3')
    to_approx = {c1: 0.14, c2: 0.3, c4: 0.12, m2: 28} 
    initial_beta = {c3: 0.1, m1: 10, m3: 21}  # initial approximation for β = (c3, m1, m3)

    result = approximate(input_data, to_approx, [c3, m1, m3], initial_beta, eps=1e-6)
    print("Approximated values: ", result)
