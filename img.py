
import time
import numpy as np
from DMD import object_extraction, compute_newD, rDMD_batch
from rDMDio import showimages, readgt, loadimgs


# 3000 is okay
imgNo = 200
batchsize = 100
A, X, Y, snapshots, x_pix, y_pix = loadimgs(num = imgNo, filepath='D:/input/')
rank = 98
p = rank
q = 5


#############################################
# rdmd & backgorund/foreground extraction

start = time.clock()
# for threshold
# 0.001 is okay, 0.01 is too big, 0.0001 is too small

# phi, B, V1, V2, V3 = object_extraction(X,Y,A,rank,p, threshold= 0.009)
output, parameters = rDMD_batch(X,Y,A,rank)
# Background = compute_newD(phi, B, V1)
# Object = compute_newD(phi, B, V2)
# Full = compute_newD(phi, B, V3)
#
# showimages(A = Object.real,x_pix = x_pix,y_pix = y_pix,num= imgNo, filepath='D:/objects/')
# showimages(A = Background.real,x_pix = x_pix,y_pix = y_pix,num= imgNo, filepath='D:/background/')
# showimages(A = Full.real,x_pix = x_pix,y_pix = y_pix,num= imgNo, filepath='D:/output/')
end = time.clock()
print("rdmd:" + str(end-start))

#####################
# error computation

# start2 = time.clock()
# B= readgt(num=imgNo, filepath='D:/groundtruth/')
# Error = B - Object.real
# print(Error.shape)
# error = np.sum(np.sum(Error))/x_pix/y_pix/imgNo
# print("error")
# print(error)
# # showimages(A = Error,x_pix = x_pix,y_pix = y_pix,num= imgNo, filepath='D:/error/')
# end2 = time.clock()
# print("error estimation time:" + str(end2-start2))