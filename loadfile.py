import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt  # plt 用于显示图片

def loadimgs(num = 100):
    snapshots = [
            np.array(cv2.imread('D:/input/in00{:04d}.jpg'.format(i),0),dtype='uint32')
            for i in range(1,num +1)
        ]
    # declare A as the (m,n) matrix which contains the whole images
    A = np.zeros([snapshots[0].shape[0] * snapshots[0].shape[1], len(snapshots)])
    n = len(snapshots)
    for i in range(0,n):
        A[:,i] = snapshots[i].reshape((snapshots[i].shape[0] * snapshots[i].shape[1],1))[0]
        # print(snapshot.shape)
        # print(snapshot)
        # plt.imshow(snapshot)
        # plt.show()

    X = A[:,range(len(snapshots)-1)]
    Y = A[:,range(1,len(snapshots))]
    x_pix = snapshots[0].shape[0]
    y_pix = snapshots[0].shape[1]

    return A, X, Y, snapshots, x_pix, y_pix

def showimages(A, x_pix, y_pix, num = 100):
    print(A.shape)
    snapshots = [
        A[:,i].reshape((x_pix,y_pix))
        for i in range(num)
    ]
    # print(len(snapshots))
    # print(snapshots[0].shape)
    print(snapshots[0])
    print(snapshots[3])
    for j in range(A.shape[1]):
        # print(snapshot)
        img = Image.fromarray(snapshots[j])
        # 转换成灰度图
        # img = img.covert('L')
        # 可以调用Image库下的函数了，比如show()
        img.show()




