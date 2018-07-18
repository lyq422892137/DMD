from loadfile import loadimgs

# from svd import svd_newMatrix
from pydmd import FbDMD
import matplotlib.pyplot as plt
# import numpy as np
# import scipy.integrate
import time
from DMD import rdmd
from DMD import compute_newD
from loadfile import showimages

# 3000 is okay
imgNo = 400
A, X, Y, snapshots, x_pix, y_pix = loadimgs(imgNo)
# batchsize =
rank = 190
p = rank
q = 5

start = time.clock()
phi, B, V= rdmd(X,Y,A,rank,p)
D_new = compute_newD(phi, B,V)
showimages(D_new.real,x_pix,y_pix,imgNo)
end = time.clock()
print("rdmd:" + str(end-start))
