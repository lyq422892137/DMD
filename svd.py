import numpy as np

np.random.seed(7)

m = 20
n = 3
A = np.random.random((m,n))


# to calculate VT, the (n,n) left singular vector
# S (m,n) diagonal matrix by sigma
# U the right singular vector
# a is the eigenvalue of A*A.T
# v is the eigenvector of A*A.T

U, sigma, VT = np.linalg.svd(np.dot(A.T,A))


# if m >= n, add m-n columns of 0s to S
S = np.diag(sigma)
if m >=n:
    S = np.row_stack((S, np.zeros((m-n,n))))
else:
    print("m is smaller than n!")

print(S.shape)

# calculate new A
A_new = np.dot(np.dot(S,U), VT)
print(A_new.shape)

D = np.sum(A - A_new)
print(D)

#################
# try jpg
from loadfile import loaddata

img = loaddata(1)


import cv2
img2 = cv2.imread('D:/input/in00{:04d}.jpg', 1)
img_pix = np.array(img, dtype='uint8')
print(img_pix.shape)
print(img_pix)
