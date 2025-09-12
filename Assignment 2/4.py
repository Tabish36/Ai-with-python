import numpy as np

A = np.array([[1,2,3],
              [0,1,4],
              [5,6,0]])

A_inv = np.linalg.inv(A)
print("Inverse of A:\n", A_inv)

I1 = np.dot(A, A_inv)
I2 = np.dot(A_inv, A)

print("A * A^-1:\n", I1)
print("A^-1 * A:\n", I2)
