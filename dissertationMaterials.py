
import time
from DMD import object_extraction, compute_newD
from rDMDio import loadimgs,  downloadImgs
import gc
import numpy as np
from rDMDio import ImgstoVideo, showimages
import cv2

gc.collect()
# 3000 is okay, > 20 is better
imgNo = 2000
# imgNo = 1700
A, X, Y, snapshots, x_pix, y_pix = loadimgs(num=imgNo, filepath='D:/input/')

A= A.reshape((x_pix*y_pix,imgNo))
print(A.shape)
cv2.imwrite("D://frames.png", A)
