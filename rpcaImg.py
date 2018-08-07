
import time
from rDMDio import loadimgs,  downloadImgs
import gc
import numpy as np
from rpca import robust_pca
import cv2


gc.collect()
# 3000 is okay, > 20 is better
imgNo = 100
# imgNo = 1700
batchsize = 100
rank = 448
A, X, Y, snapshots, x_pix, y_pix = loadimgs(num=imgNo, filepath='D:/input/')
n = A.shape[1]
m = A.shape[0]
p = rank
q = 5



#############################################
# rdmd & backgorund/foreground extraction

start = time.clock()
L, S = robust_pca(A)
times = int(n/batchsize)
M = {}
Dstart = 0
Dend = batchsize
subStart = 0
subEnd = batchsize - 1
rank_new = int((rank + p) * batchsize / n)
errors = 0

for i in range(imgNo):
    print("round " + str(i) + ":")
    downloadImgs(L, S, A, x_pix=x_pix, y_pix=y_pix, num=imgNo,
                 backpath='D:/background/', objpath='D:/objects/', fullpath='D:/output/', flag=i)
    gc.collect()


end = time.clock()
print("rdmd:" + str(end-start))

gc.collect()
