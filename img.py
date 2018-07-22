
import time
from DMD import object_extraction, compute_newD
from rDMDio import loadimgs,  downloadImgs
import gc
import numpy as np


gc.collect()
# 3000 is okay, > 20 is better
imgNo = 1700
batchsize = 100
threshold = 0.001
A, X, Y, snapshots, x_pix, y_pix = loadimgs(num=imgNo, filepath='D:/input/')
rank = 848
n = A.shape[1]
m = A.shape[0]
p = rank
q = 5

#############################################
# rdmd & backgorund/foreground extraction

start = time.clock()

times = int(n/batchsize)
M = {}
Dstart = 0
Dend = batchsize
subStart = 0
subEnd = batchsize - 1
rank_new = int((rank + p) * batchsize / n)
errors = 0

if np.mod(n, batchsize) !=0:
    print("A")
else:
    for i in range(times):

        print("round " + str(i) + ":")

        M["D" + str(i)] = A[:, Dstart:Dend]
        M["X" + str(i)] = X[:, subStart:subEnd]
        M["Y" + str(i)] = Y[:, subStart:subEnd]

        phi, B, V1, V2, V3 = object_extraction(X=M["X" + str(i)], Y=M["Y" + str(i)],
                                                                              D=M["D" + str(i)],
                                                                              rank=rank_new, p=0, q=q,
                                                                              threshold=threshold)
        Background = compute_newD(phi, B, V1)
        Objects = compute_newD(phi, B, V2)
        Full = compute_newD(phi, B, V3)

        downloadImgs(Background.real, Objects.real, Full.real, x_pix=x_pix, y_pix=y_pix, num=batchsize,
                     backpath='D:/background/', objpath='D:/objects/', fullpath='D:/output/', flag=i * batchsize)

        del phi, B, V1, V2, V3, Background, Full, Objects
        gc.collect()

        Dstart = Dend
        Dend = Dend + batchsize
        subStart = subEnd
        subEnd = subEnd + batchsize

end = time.clock()
print("rdmd:" + str(end-start))

gc.collect()
