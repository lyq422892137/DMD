
import time
from DMD import object_extraction, compute_newD
from rDMDio import loadimgs,  downloadImgs
import gc
import numpy as np
from rDMDio import ImgstoVideo
from rpca import robust_pca



# 3000 is okay, > 20 is better

imgNo = 700
# imgNo = 200
batchsize = 100
A, X, Y, snapshots, x_pix, y_pix = loadimgs(num=imgNo, filepath='D:/input_hw/')
del snapshots
gc.collect()
rank = imgNo/2-1
# rank = 49
n = A.shape[1]
m = A.shape[0]
p = rank
q = 5

#############################################
# rdmd & backgorund/foreground extraction

start = time.clock()
Dstart = 0
Dend = batchsize
times = int(n/batchsize)
M = {}
subStart = 0
subEnd = batchsize - 1
rank_new = int((rank + p) * batchsize / n)
errors = 0

if np.mod(n, batchsize) !=0:
    print("A")
else:
    for i in range(times):

        print("round " + str(i) + ":")

        M["X" + str(i)] = X[:, subStart:subEnd]
        M["Y" + str(i)] = Y[:, subStart:subEnd]
        M["D" + str(i)] = A[:, Dstart:Dend]

        phi, B, V1, V2, V3 = object_extraction(X=M["X" + str(i)], Y=M["Y" + str(i)],
                                                                              D=M["D" + str(i)],
                                                                              rank=rank_new, p=0, q=q, imgNo=imgNo)
        # L, S = robust_pca(M["D" + str(i)])
        # S = S * np.power(10, 4.2)
        #
        # Background = compute_newD(phi, B, V1)
        # Objects = compute_newD(phi, B, V2)
        # Full = compute_newD(phi, B, V3)
        #
        # downloadImgs(Background.real, Objects.real, Full.real, x_pix=x_pix, y_pix=y_pix, num=batchsize,
        #              backpath='D:/background/', objpath='D:/objects/', fullpath='D:/output/', flag=i * batchsize)

        # downloadImgs(L, S, M["D" + str(i)], x_pix=x_pix, y_pix=y_pix, num=batchsize,
        #              backpath='D:/background/', objpath='D:/objects/', fullpath='D:/output/', flag=i * batchsize)

        # del phi, B, V1, V2, V3, Background, Full, Objects
        del phi, B, V1, V2, V3
        # del L, S
        gc.collect()

        subStart = subEnd
        subEnd = subEnd + batchsize

end = time.clock()
print("rdmd:" + str(end-start))

gc.collect()

ImgstoVideo("D:/objects/",'D:/videos/video3.avi',x_pix,y_pix)