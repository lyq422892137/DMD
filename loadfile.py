import numpy as np
import cv2

import matplotlib.pyplot as plt  # plt 用于显示图片

##################################
# def loaddata(num = 100):
#     snapshots = [
#         np.array(np.array(cv2.imread('D:/input/in00{:04d}.jpg'.format(i),0)),dtype='uint32').reshape((320*240,1))
#         for i in range(1,num +1)
#     ]
#
#     for i, snapshot in enumerate(snapshots, start=1):
#         print(snapshot.shape)
#         # plt.imshow(snapshot)
#         # plt.show()
#
#     print(str(len(snapshots)) + " images")
#
#     snapshots = np.array(snapshots).reshape((snapshots,1))
#     return snapshots

def loadimgs(num = 100):
    snapshots = [
            np.array(cv2.imread('D:/input/in00{:04d}.jpg'.format(i),0),dtype='uint32')
            for i in range(1,num +1)
        ]
    # declare A as the (m,n) matrix which contains the whole images
    A = np.zeros([snapshots[0].shape[0] * snapshots[0].shape[1], len(snapshots)])
    for i, snapshot in enumerate(snapshots, start=1):
        A[:,i-1] = snapshot.reshape((snapshot.shape[0] * snapshot.shape[1],1))[0]
        # print(snapshot.shape)
        # plt.imshow(snapshot)
        # plt.show()

    X = A[:,0:A.shape[1]-1]
    Y = A[:,1:A.shape[1]]

    return A, X, Y





