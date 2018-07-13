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
    for i, snapshot in enumerate(snapshots, start=1):
        A[:,i-1] = snapshot.reshape((snapshot.shape[0] * snapshot.shape[1],1))[0]
        # print(snapshot.shape)
        # plt.imshow(snapshot)
        # plt.show()

    X = A[:,0:A.shape[1]-1]
    Y = A[:,1:A.shape[1]]

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

    for i, snapshot in enumerate(snapshots, start=1):
        img = Image.fromarray(snapshot)
        # 转换成灰度图
        # img = img.covert('L')
        # 可以调用Image库下的函数了，比如show()
        img.show()




