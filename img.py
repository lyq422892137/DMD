
import time
from DMD import rDMD_batch
from rDMDio import loadimgs, showImgs_batch_error, showImgs_batch
import gc


gc.collect()

# 3000 is okay, > 20 is better
imgNo = 1700
batchsize = 100
threshold = 0.001
A, X, Y, snapshots, x_pix, y_pix = loadimgs(num=imgNo, filepath='D:/input/')
rank = 198
n = A.shape[1]
p = rank
q = 5

#############################################
# rdmd & backgorund/foreground extraction

start = time.clock()

output, parameters = rDMD_batch(X=X, Y=Y, D=A, rank=rank, threshold=threshold, p=p, q=q, batchsize=batchsize)
del parameters
del X, Y, A, rank, threshold, p, q, batchsize
gc.collect()

# errors = showImgs_batch_error(matrices=output, n=A.shape[1], x_pix=x_pix, y_pix=y_pix)
# print("Total errors: " + str(errors))

showImgs_batch(matrices=output, n=n, x_pix=x_pix, y_pix=y_pix)

end = time.clock()
print("rdmd:" + str(end-start))

del output
gc.collect()
