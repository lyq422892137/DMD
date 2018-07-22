
import time
import numpy as np
from DMD import object_extraction, compute_newD, rDMD_batch
from rDMDio import showimages, readgt, loadimgs, showImgs_batch, downloadImgs


# 3000 is okay, > 20 is better
imgNo = 1700
batchsize = 100
threshold = 0.001
A, X, Y, snapshots, x_pix, y_pix = loadimgs(num=imgNo, filepath='D:/input/')
rank = 849
p = rank
q = 5

#############################################
# rdmd & backgorund/foreground extraction

start = time.clock()

output, parameters = rDMD_batch(X=X, Y=Y, D=A, rank=rank, threshold=threshold, p=p, q=q, batchsize=batchsize)
showImgs_batch(matrices=output, n=A.shape[1], x_pix=x_pix, y_pix=y_pix)
# print("Total errors: " + str(errors))

end = time.clock()
print("rdmd:" + str(end-start))
