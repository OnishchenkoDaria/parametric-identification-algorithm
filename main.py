import numpy as np

def read_numbers_from_file(filename):
    with open(filename, 'r') as file:
        content = file.read()
        numbers = list(map(float, content.split()))
    return numbers

#setting b vectors fir iterations
def set_B0():
    B0 = np.zeros((3, 1))
    B0[:, 0] = [0.1, 10, 21]
    return B0

B = np.zeros((3, 1))

def set_B(c3, m1, m3):
    B[:, 0] = [c3, m1, m3]


#matrix of the mathimatical problem
def model_mtrx(c1, c2, c3, c4, m1, m2, m3):
    # c3, m1, m3 - remain uninitialized, need to find
    A = [[0, 1, 0, 0, 0, 0],
        [ -(c1 + c2)/m1, 0, c2/m1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [c2/m2, 0, -(c2 + c3)/ m2, 0, c3/m3, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, c3/m3, 0, -(c3+c4)/m3, 0]]
    return A


#derivatives with respect to variable
def get_derivative(y_vec, b_vec, b_values, h=1e-5):

    derivs = [] 

    for y_i in y_vec:
        #set primary y0 
        y_original = y_i(b_values) 
        row = []
        
        #for each b_i (as an index)
        for b_i in b_vec:
            #apdate b_i with step
            temp = b_values[b_i]
            b_values[b_i] = temp + h
            y_step = y_i(b_values)
            
            #calculate derivative using formula
            d = (y_step - y_original) / h
            row.append(d)

            #set to default
            b_values[b_i] = temp
        
        derivs.append(row)

    return derivs

#partial derivatives of the vector Ay with respect to the vector beta
""" def compute_A_deriv_beta(Ay, beta, h=1e-5):
    
    num_rows = len(Ay)
    num_cols = len(beta)
    derivative_matrix = np.zeros((num_rows, num_cols))

    for i in range(num_rows):
        for j in range(num_cols):
            
            #copy data
            beta_step = list(beta)
            beta_step[j] += h  #step of finite difference
            
            # aproximate the derivative using the finite difference method
            f_x_plus_h = Ay[i](beta_step)  #Ay evaluated at beta + h for j-th element
            f_x = Ay[i](beta)              #Ay evaluated at the original beta
            
            #finite difference and store it in the matrix
            derivative_matrix[i][j] = (f_x_plus_h - f_x) / h
    
    return derivative_matrix """

def mat_vec_mult(matrix, vector):
        #multiplies a matrix with a vector
        result = []
        for row in matrix:
            result.append(sum(x * v for x, v in zip(row, vector)))
        return result

#using runge-kutta
def get_u_matr(a_matr, b_matr, u_matr, h):
    #convert b_matr to an array
    b_arrayed = b_matr.tolist()    
    
    #method formula
    k1 = h * (mat_vec_mult(a_matr, u_matr) + b_arrayed)
    k2 = h * (mat_vec_mult(a_matr, [u + k / 2 for u, k in zip(u_matr, k1)]) + b_arrayed)
    k3 = h * (mat_vec_mult(a_matr, [u + k / 2 for u, k in zip(u_matr, k2)]) + b_arrayed)
    k4 = h * (mat_vec_mult(a_matr, [u + k for u, k in zip(u_matr, k3)]) + b_arrayed)

    #return the updated u_matr
    return [u + (k1_val + 2 * k2_val + 2 * k3_val + k4_val) / 6 for u, k1_val, k2_val, k3_val, k4_val in zip(u_matr, k1, k2, k3, k4)]

def get_y(a_matr, y_cur, h):

    #calculate k1, k2, k3, k4 using matrix-vector multiplication manually
    k1 = h * mat_vec_mult(a_matr, y_cur)
    k2 = h * mat_vec_mult(a_matr, [y + k1_elem / 2 for y, k1_elem in zip(y_cur, k1)])
    k3 = h * mat_vec_mult(a_matr, [y + k2_elem / 2 for y, k2_elem in zip(y_cur, k2)])
    k4 = h * mat_vec_mult(a_matr, [y + k3_elem for y, k3_elem in zip(y_cur, k3)])

    #return the updated y_cur]
    return [y + (k1_elem + 2 * k2_elem + 2 * k3_elem + k4_elem) / 6 for y, k1_elem, k2_elem, k3_elem, k4_elem in zip(y_cur, k1, k2, k3, k4)]

if __name__ == "__main__":
    numbers = read_numbers_from_file("y1.txt")
    print(numbers[0])
    A = model_mtrx(1, 2, 3, 4, 4, 5, 6)
    print(A)