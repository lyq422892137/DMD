
import time
from DMD import object_extraction, compute_newD, errorComputation
from rDMDio import loadimgs,  downloadImgs, showimages
import gc
import numpy as np
from rDMDio import ImgstoVideo, showimages, readgt
import cv2

gc.collect()

# 3000 is okay, > 20 is better

# imgNo = 1700
# A, X, Y, snapshots, x_pix, y_pix = loadimgs(num=imgNo, filepath='D:/input/')
#
# A= A.reshape((x_pix*y_pix,imgNo))
# print(A.shape)
# cv2.imwrite("D://frames.png", A)



# creat masks
imgNo = 300
maskthresh= 40

newA, newX, newY, newsnapshots, newx_pix, newy_pix = loadimgs(num=imgNo, filepath='D:/objects/')
# print(newA)
maskflag = np.greater(newA,maskthresh)
masks = maskflag*255

showimages(A=masks, x_pix=newx_pix,y_pix=newy_pix, filepath="D:/masks/",flag=0,num=imgNo)
# masks = masks1[:,400:1700]

gc.collect()
# print(masks)
G = readgt("D:/groundtruth_subway/", imgNo)

# G=G1[:,400:1700]
# del G1
gc.collect()

Error = abs(sum(G,0)) - abs(sum(masks,0))
error = sum(Error) / newx_pix / newy_pix/imgNo
print(error)
del Error, error
gc.collect()
#
# print(len(sum(E,0)))
# Error = sum(E[:,470:1699],0)/newy_pix/newx_pix

# encoding=utf-8
import matplotlib.pyplot as plt
# x = range(470, len(Error)+470)
# y = Error
# plt.plot(x, y)
# plt.margins(0)
# plt.subplots_adjust(bottom=0.15)
# plt.xlabel("image No.")
# plt.ylabel("Error values")
# plt.title("Average Errors for each image")

# plt.show()

# masks = masks[:,400:]
# G= G[:,400:]

zero_array = np.zeros(masks.shape)
roi_array=np.empty(G.shape,dtype='uint8')
roi_array.fill(85)
fgd_bw=np.greater(masks, zero_array)
print(fgd_bw)
truth_bw=np.logical_not(np.logical_or(np.equal(G,zero_array),np.equal(G,roi_array)))

print(np.count_nonzero(truth_bw))
TP=np.count_nonzero(np.multiply(fgd_bw,truth_bw))
FP=np.count_nonzero(np.multiply(np.logical_not(truth_bw),fgd_bw))
TN=np.count_nonzero(np.multiply(np.logical_not(truth_bw),np.logical_not(fgd_bw)))
FN=np.count_nonzero(np.multiply(np.logical_not(fgd_bw),truth_bw))

print(TP, FP, TN, FN)

misclassrate = (FP+FN)/(TP+TN+FP+FN)
recall=TP/(TP+FN)
precision=TP/(TP+FP)
fs=2*(precision*recall)/(precision+recall)
specificity = TN/(TN+FP)

print(fs,  precision, recall,1-misclassrate, specificity)