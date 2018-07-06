import numpy as np
import cv2

import matplotlib.pyplot as plt  # plt 用于显示图片

##################################
def loaddata(num = 100):
    snapshots = [
        np.array(np.array(cv2.imread('D:/input/in00{:04d}.jpg'.format(i),0)),dtype='uint32').reshape((320*240,1))
        for i in range(1,num +1)
    ]

    for i, snapshot in enumerate(snapshots, start=1):
        print(snapshot.shape)
        # plt.imshow(snapshot)
        # plt.show()

    print(str(len(snapshots)) + " images")

    snapshots = np.array(snapshots).reshape((snapshots,1))
    return snapshots

