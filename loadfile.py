import numpy as np
import cv2

def loadimgs(num = 100):
    snapshots = [
            np.array(cv2.imread('D:/input/in00{:04d}.jpg'.format(i),0),dtype='uint8')
            # np.array(cv2.imread('/cs/home/yl90/Downloads/corridor/input/in00{:04d}.jpg'.format(i), 0), dtype='uint32')
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
        cv2.imwrite(filepath+"/in00{:04d}.png".format(i+1), snapshots2[i])



