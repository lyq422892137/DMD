from rDMDio import loadimgs

import time
import numpy as np
from DMD import rdmd
from DMD import object_extraction
from DMD import compute_newD
from rDMDio import showimages
from rDMDio import readgt


# 3000 is okay
imgNo = 200
A, X, Y, snapshots, x_pix, y_pix = loadimgs(num = imgNo, filepath='D:/input/')
# batchsize =
rank = 98
p = rank
q = 5


#############################################
# rdmd & backgorund/foreground extraction

start = time.clock()
# for threshold
# 0.001 is okay, 0.01 is too big, 0.0001 is too small

phi, B, V1, V2, V3 = object_extraction(X,Y,A,rank,p, threshold= 0.003)
Background = compute_newD(phi, B, V1)
Object = compute_newD(phi, B, V2)
Full = compute_newD(phi, B, V3)

showimages(A = Object.real,x_pix = x_pix,y_pix = y_pix,num= imgNo, filepath='D:/objects/')
showimages(A = Background.real,x_pix = x_pix,y_pix = y_pix,num= imgNo, filepath='D:/background/')
showimages(A = Full.real,x_pix = x_pix,y_pix = y_pix,num= imgNo, filepath='D:/output/')

end = time.clock()
print("rdmd:" + str(end-start))

start2 = time.clock()
B= readgt(num=imgNo, filepath='D:/groundtruth/')
Error = B - Object.real
print(Error.shape)
error = np.sum(np.sum(Error))/x_pix/y_pix/imgNo
print(error.shape)
print("error")
print(error)
# showimages(A = Error,x_pix = x_pix,y_pix = y_pix,num= imgNo, filepath='D:/error/')
end2 = time.clock()
print("error estimation time:" + str(end2-start2))