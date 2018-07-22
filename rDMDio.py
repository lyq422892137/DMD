import numpy as np
import cv2
import gc


def loadimgs(filepath, num = 100):
    snapshots = [
            np.array(cv2.imread(filepath + 'in00{:04d}.jpg'.format(i),0),dtype='uint8')
            for i in range(1,num +1)
        ]

    # declare A as the (m,n) matrix which contains the whole images
    A = np.zeros([snapshots[0].shape[0] * snapshots[0].shape[1], len(snapshots)])
    n = len(snapshots)
    for i in range(n):
        A[:,i] = snapshots[i].reshape((snapshots[i].shape[0] * snapshots[i].shape[1],))

    X = A[:,0:len(snapshots)-1]
    Y = A[:,1:len(snapshots)]

    x_pix = snapshots[0].shape[0]
    y_pix = snapshots[0].shape[1]

    return A, X, Y, snapshots, x_pix, y_pix


def showimages(A, x_pix, y_pix, filepath, flag, num = 100):
    batchsize = A.shape[1]
    snapshots2 = [
        A[:, i].reshape((x_pix, y_pix))
        for i in range(num)
    ]
    n = 0
    for i in range(flag, flag + batchsize):
        cv2.imwrite(filepath+"in00{:04d}.png".format(i+1), snapshots2[n])
        n = n + 1
    del A, filepath, snapshots2
    gc.collect()

def readgt(filepath, num = 100):
    snapshots = [
        np.array(cv2.imread(filepath + '\gt00{:04d}.png'.format(i), 0), dtype='uint8')
        for i in range(1, num + 1)
    ]

    # declare B as the (m,n) matrix which contains the whole images
    G = np.zeros([snapshots[0].shape[0] * snapshots[0].shape[1], len(snapshots)])
    n = len(snapshots)
    for i in range(n):
        G[:, i] = snapshots[i].reshape((snapshots[i].shape[0] * snapshots[i].shape[1],))
    return G


def downloadImgs(background, objects, full, x_pix, y_pix, num, backpath, objpath, fullpath, flag):
    showimages(A=objects, x_pix=x_pix, y_pix=y_pix, num=num, filepath=objpath, flag=flag)
    showimages(A=background, x_pix=x_pix, y_pix=y_pix, num=num, filepath=backpath, flag=flag)
    showimages(A=full, x_pix=x_pix, y_pix=y_pix, num=num, filepath=fullpath, flag=flag)











