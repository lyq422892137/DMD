
import time
from rDMDio import loadimgs,  downloadImgs
import gc
import numpy as np
from rpca import robust_pca
import cv2


gc.collect()
imgNo = 700
# imgNo = 1700
A, X, Y, snapshots, x_pix, y_pix = loadimgs(num=imgNo, filepath='D:/input_hw/')
n = A.shape[1]
m = A.shape[0]


#############################################
# rdmd & backgorund/foreground extraction

start = time.clock()
L, S = robust_pca(A)

print(L)
print(S.shape)
print(S)
S = S * np.power(10,7)
# print(S.reshape((2240,100)))
# S= S * np.power(10,4.5)
# print(S.reshape((2240,100)))

# Dstart = 0
# Dend = batchsize
# subStart = 0
# subEnd = batchsize - 1
# rank_new = int((rank + p) * batchsize / n)
# errors = 0

# for i in range(imgNo):
#     print("round " + str(i) + ":")

end = time.clock()
print("rdmd:" + str(end-start))

# downloadImgs(L, S, A, x_pix=x_pix, y_pix=y_pix, num=imgNo,
#                  backpath='D:/background/', objpath='D:/objects/', fullpath='D:/output/', flag=0)
    # print(i)
gc.collect()

