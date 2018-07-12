from loadfile import loadimgs
from svd import rsvd
from svd import cal_svd
from svd import svd_newMatrix
from pydmd import DMD
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from svd import rDMD
import time

imgNo = 1000
A, X, Y, snapshots = loadimgs(imgNo)
# batchsize =
rank = 5
###################################################

print(X.shape)
print(Y.shape)
print((X==Y).all())
print(A.shape)


#############
# DMD packages
# dmd = DMD(svd_rank=rank)
# dmd.fit(A.T)
#
# for eig in dmd.eigs:
#     print('Eigenvalue {}: distance from unit circle {}'.format(eig, np.abs(eig.imag**2 + eig.real**2 - 1)))
# dmd.plot_eigs(show_axes= True, show_unit_circle= True)

# #######################
# by rsvd, we cannot compute new matrix directly even the number of images are 2, batches are needed
start1 = time.clock()

Ux, sigmax, Ax = rsvd(A = X, rank = rank, p = 0, q = 1)
end1 = time.clock()

print("rsvd:" + str(end1-start1))

########################
# by svd, we can compute new matrix directly with 1000 images
start2 = time.clock()
U2, sigma2, V2 = cal_svd(X, rank)
end2 = time.clock()

print("svd:" + str(end2-start2))