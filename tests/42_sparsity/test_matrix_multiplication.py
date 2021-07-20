import time
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

n = 20

A = np.random.rand(n,n)
A[A < 0.99] = 0
x = np.random.rand(n)

# plt.matshow(A)
# plt.show()

starttime = time.time()
y = A @ x
print(f'Runtime np.dot: {(time.time() - starttime) * 1000:.3f} ms')

A_bsr = sp.bsr_matrix(A)
starttime = time.time()
y = A_bsr.dot(x)
print(f'Runtime BSR Sparse: {(time.time() - starttime) * 1000:.3f} ms')

A_coo = sp.coo_matrix(A)
starttime = time.time()
y = A_coo.dot(x)
print(f'Runtime COO Sparse: {(time.time() - starttime) * 1000:.3f} ms')

A_csc = sp.csc_matrix(A)
starttime = time.time()
y = A_csc.dot(x)
print(f'Runtime CSC Sparse: {(time.time() - starttime) * 1000:.3f} ms')

A_csr = sp.csr_matrix(A)
starttime = time.time()
y = A_csr.dot(x)
print(f'Runtime CSR Sparse: {(time.time() - starttime) * 1000:.3f} ms')

# A_dia = sp.dia_matrix(A)
# starttime = time.time()
# y = A_dia.dot(x)
# print(f'Runtime DIA Sparse: {(time.time() - starttime) * 1000:.3f} ms')

# A_dok = sp.dok_matrix(A)
# starttime = time.time()
# y = A_dok.dot(x)
# print(f'Runtime DOK Sparse: {(time.time() - starttime) * 1000:.3} ms')

A_lil = sp.lil_matrix(A)
starttime = time.time()
y = A_lil.dot(x)
print(f'Runtime LIL Sparse: {(time.time() - starttime) * 1000:.3f} ms')

