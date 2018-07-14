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
    m = snapshots[0].shape[0]
    print("342342342")
    print(snapshots[0].reshape((snapshots[0].shape[0] * snapshots[0].shape[1],1)))
    for i in range(n):
        A[:,i] = snapshots[i].reshape((snapshots[i].shape[0] * snapshots[i].shape[1],))
        # print(snapshot.shape)
        # print(snapshot)
        # plt.imshow(snapshot)
        # plt.show()

    X = A[:,range(len(snapshots)-1)]
    Y = A[:,range(1,len(snapshots))]

    x_pix = snapshots[0].shape[0]
    y_pix = snapshots[0].shape[1]

    snapshots2 = [
        A[:, i].reshape((x_pix, y_pix))
        for i in range(num)
    ]

    print(type(snapshots))
    print(type(snapshots2))
    # img = Image.fromarray(snapshots2[0])
    cv2.imwrite("d://1.png",snapshots[0])
    print(snapshots[0].shape)
    print(snapshots[0])
    print("-----------")
    cv2.imwrite("d://2.png", snapshots2[0])
    print(snapshots2[0].shape)
    print(snapshots2[0])

    # img2 = Image.fromarray(snapshots2[3])
    # img2.show()


    return A, X, Y, snapshots, x_pix, y_pix

def showimages(A, x_pix, y_pix, num = 100):
    print(A.shape)
    snapshots = [
        A[:,i].reshape((x_pix,y_pix))
        for i in range(num)
    ]
    # print(len(snapshots))
    # print(snapshots[0].shape)
    print("snap[0]:"+str(snapshots[0]))
    print("snap[3]:"+str(snapshots[3]))
    # img = Image.fromarray(snapshots[0])
    # img.show()
    #
    # img2 = Image.fromarray(snapshots[3])
    # img2.show()
    # for j in range(A.shape[1]):
    #     print(snapshots[j])
        # img = Image.fromarray(snapshots[j])
        # img.show()




