from numpy import *
from svd import cal_svd
from svd import rsvd
from svd import svd_newMatrix
from svd import rsvd_newMatrix

random.seed(7)
m = 320*240
n = 2
rank = 1
A = random.random((m,n))

# svd
U, sigma, V = cal_svd(A,rank)
A_new, D = svd_newMatrix(A, U, sigma, V, rank)
# print(A)
# print("-----------------------")
# print(A_new.shape)


# rsvd
U2, sigma2, V2 = rsvd(A,rank)
A_new2, D2 = rsvd_newMatrix(A, U2, sigma2, V2, rank)
# print(A)
# print("-----------------------")
# print(A_new2)



