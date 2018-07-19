from loadfile import loadimgs

import time
from DMD import rdmd
from DMD import object_extraction
from DMD import compute_newD
from loadfile import showimages

# 3000 is okay
imgNo = 50
A, X, Y, snapshots, x_pix, y_pix = loadimgs(imgNo)
# batchsize =
rank = 24
p = rank
q = 5

start = time.clock()
phi, B, V1, V2, V3 = object_extraction(X,Y,A,rank,p)
Background = compute_newD(phi, B, V1)
Object = compute_newD(phi, B, V2)
Full = compute_newD(phi, B, V3)

showimages(A = Object.real,x_pix = x_pix,y_pix = y_pix,num= imgNo, filepath='D:/objects')
showimages(A = Background.real,x_pix = x_pix,y_pix = y_pix,num= imgNo, filepath='D:/background')
showimages(A = Full.real,x_pix = x_pix,y_pix = y_pix,num= imgNo, filepath='D:/output')

end = time.clock()
print("rdmd:" + str(end-start))
