
import time
import numpy as np
from DMD import object_extraction, compute_newD, rDMD_batch
from rDMDio import showimages, readgt, loadimgs, seperateMatrix, downloadImgs


# 3000 is okay, > 20 is better
imgNo = 100
batchsize = 100
threshold = 0.001
A, X, Y, snapshots, x_pix, y_pix = loadimgs(num = imgNo, filepath='D:/input/')
rank = 49
p = rank
q = 5

#############################################
# rdmd & backgorund/foreground extraction

start = time.clock()

output, parameters = rDMD_batch(X=X, Y=Y, D=A, rank=rank, threshold=threshold, p=p, q=q, batchsize=batchsize)
Background, Objects, Full = seperateMatrix(output, A.shape[1], x_pix=x_pix, y_pix=y_pix)

# phi, B, V1, V2, V3 = object_extraction(X=X, Y=Y, D=A, rank=rank, threshold=threshold, p=p, q=q)
# Background = compute_newD(phi,B,V1)
# Objects = compute_newD(phi,B,V2)
# Full = compute_newD(phi,B,V3)


end = time.clock()
print("rdmd:" + str(end-start))

#####################
# error computation
start2 = time.clock()
B= readgt(num=imgNo, filepath='D:/groundtruth/')
Error = B - Objects.real
error = np.sum(np.sum(Error))/x_pix/y_pix/imgNo
print("error: " + str(error))
showimages(A = Error,x_pix = x_pix,y_pix = y_pix,num= imgNo, filepath='D:/error/')
end2 = time.clock()
print("error estimation time:" + str(end2-start2))