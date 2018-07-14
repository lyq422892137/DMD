from loadfile import loadimgs
from svd import rsvd
from svd import cal_svd
# from svd import svd_newMatrix
# from pydmd import DMD
# import matplotlib.pyplot as plt
# import numpy as np
# import scipy.integrate
import time
from DMD import rdmd
# from DMD import computeImags
from DMD import compute_newD
# from DMD import separateOmega
# from DMD import compute_background
# from DMD import compute_foreground
from loadfile import showimages



# 3000 is okay
imgNo = 20
A, X, Y, snapshots, x_pix, y_pix = loadimgs(imgNo)
# batchsize =
rank = 5
p = 0
###################################################
# print(Y)
# print(X.shape)
# print(Y.shape)
print((X[:,0]==Y[:,0]).all())
print((X[:,1]==Y[:,0]).all())
print((X==Y).all())
# print(X)
# print(A.shape)


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
# start1 = time.clock()
#
# Ux, sigmax, Ax = rsvd(A = X, rank = rank, p = 0, q = 1)
# end1 = time.clock()
#
# print("rsvd:" + str(end1-start1))
#
# ########################
# # by svd, we can compute new matrix directly with 1000 images
# start2 = time.clock()
# U2, sigma2, V2 = cal_svd(X, rank)
# end2 = time.clock()
#
# print("svd:" + str(end2-start2))

# for 1000 images:
# rsvd:1.0923066157967092
# svd:189.06396405083467

start3 = time.clock()
phi, B, omega, V = rdmd(X,Y,A,rank,p)
# print(omega)
# print(B)
# print(phi)
# computeImags(omega,B,phi, A.shape[1])
D_new = compute_newD(phi, B, A.shape[1], rank+p, omega=omega, V=V)
print((D_new[:,0]==D_new[:,2]).all())
# print(D_new[10,10])
# l,s, l_count, s_count = separateOmega(omega)
# L = compute_background(D_new,s_count, A.shape[1], k ,omega,phi, B)
# print(L.shape)
# S = compute_foreground(D_new,l_count)
# print(S.shape)
# showimages(L.real,x_pix,y_pix,L.shape[1])
# showimages(S.real,x_pix,y_pix,S.shape[1])
# showimages(X,x_pix,y_pix,X.shape[1])
end3 = time.clock()
print("rdmd:" + str(end3-start3))
