import numpy as np
import cv2

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


def showimages(A, x_pix, y_pix, filepath, num = 100):
    snapshots2 = [
        A[:, i].reshape((x_pix, y_pix))
        for i in range(num)
    ]

    for i in range(num):
        cv2.imwrite(filepath+"in00{:04d}.png".format(i+1), snapshots2[i])


def readgt(filepath, num = 100):
    snapshots = [
        np.array(cv2.imread(filepath + 'gt00{:04d}.png'.format(i), 0), dtype='uint8')
        for i in range(1, num + 1)
    ]

    # declare B as the (m,n) matrix which contains the whole images
    B = np.zeros([snapshots[0].shape[0] * snapshots[0].shape[1], len(snapshots)])
    n = len(snapshots)
    for i in range(n):
        B[:, i] = snapshots[i].reshape((snapshots[i].shape[0] * snapshots[i].shape[1],))
    return B

def seperateMatrix(matrices, n):
    num = int(len(matrices)/3)
    m = matrices["background0"].shape[0]
    batchsize = matrices["background0"].shape[1]
    Background = np.zeros((m,n), dtype='complex')
    Objects = np.zeros((m,n), dtype='complex')
    Full = np.zeros((m,n), dtype='complex')
    start = 0
    end = batchsize
    for i in range(num):
        if (i == num-1) and (np.mod(n,batchsize) != 0):
            Background[:, start: ] = matrices["background" + str(i)]
            Objects[:, start:] = matrices["object" + str(i)]
            Full[:, start:] = matrices["full" + str(i)]
        else:
            Background[:, start:end]= matrices["background" + str(i)]
            Objects[:, start:end] = matrices["object" + str(i)]
            Full[:, start:end] = matrices["full" + str(i)]
            start = end
            end = end + batchsize

    return Background, Objects, Full










