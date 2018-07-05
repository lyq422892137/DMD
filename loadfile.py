from PIL import Image
import numpy as np

import matplotlib.pyplot as plt  # plt 用于显示图片

##################################
def loaddata(num = 100):
    snapshots = [
        np.array(Image.open('D:/input/in00{:04d}.jpg'.format(i)))
        for i in range(1,num +1)
    ]

    for i, snapshot in enumerate(snapshots, start=1):
        print(snapshot.shape)
        plt.imshow(snapshot)
        plt.show()

    print(len(snapshots))
    return snapshots

