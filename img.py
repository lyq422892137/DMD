from loadfile import loadimgs

import time
from DMD import rdmd
from DMD import object_extraction
from DMD import compute_newD
from loadfile import showimages

# 3000 is okay
imgNo = 20
A, X, Y, snapshots, x_pix, y_pix = loadimgs(imgNo)
# batchsize =
rank = 9
p = rank
q = 5

start = time.clock()
phi, B, V1, V2, V3 = object_extraction(X,Y,A,rank,p)
Object = compute_newD(phi, B, V1)
Background = compute_newD(phi, B, V2)
Full = compute_newD(phi, B, V3)

showimages(Object.real,x_pix,y_pix,imgNo, filepath="D:/output")

end = time.clock()
print("rdmd:" + str(end-start))
