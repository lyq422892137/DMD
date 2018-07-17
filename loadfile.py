import numpy as np
import cv2
import matplotlib.pyplot as plt  # plt 用于显示图片

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
        # print(snapshot.shape)
        # print(snapshot)
        # plt.imshow(snapshot)
        # plt.show()

    X = A[:,0:len(snapshots)-1]
    Y = A[:,1:len(snapshots)]

    x_pix = snapshots[0].shape[0]
    y_pix = snapshots[0].shape[1]

    # cv2.imwrite("d://1.png",snapshots[0])
    # print(snapshots[0].shape)
    # print(snapshots[0])
    print("-----------")


    return A, X, Y, snapshots, x_pix, y_pix

def showimages(A, x_pix, y_pix, num = 100):
    print("A.shape: " + str(A.shape))
    snapshots2 = [
        A[:, i].reshape((x_pix, y_pix))
        for i in range(num)
    ]

    for i in range(num):
        cv2.imwrite("D:/output/in00{:04d}.png".format(i+1), snapshots2[i])
    cv2.imwrite("d://1.png", snapshots2[0])
    # print(snapshots2[0].shape)
    # print(snapshots2[0])
    # print(len(snapshots))
    # print(snapshots[0].shape)
    # print("snap[0]:"+str(snapshots2[0]))
    # print("snap[1]:"+str(snapshots2[1]))
    # print("snap[2]:" + str(snapshots2[2]))
    # img = Image.fromarray(snapshots[0])
    # img.show()
    #
    # img2 = Image.fromarray(snapshots[3])
    # img2.show()
    # for j in range(A.shape[1]):
    #     print(snapshots[j])
        # img = Image.fromarray(snapshots[j])
        # img.show()




