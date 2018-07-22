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


def showImgs_batch(matrices, n, x_pix, y_pix):
    num = int(len(matrices)/3)
    m = matrices["background0"].shape[0]
    batchsize = matrices["background0"].shape[1]
    Background = np.zeros((m,batchsize), dtype='complex')
    Objects = np.zeros((m,batchsize), dtype='complex')
    Full = np.zeros((m,batchsize), dtype='complex')

    for i in range(num):
        if (i == num-1) and (np.mod(n,batchsize) != 0):
            Background[:, 0:] = matrices["background" + str(i)]
            Objects[:, 0:] = matrices["object" + str(i)]
            Full[:, 0:] = matrices["full" + str(i)]

        else:
            Background[:, 0:batchsize]= matrices["background" + str(i)]
            Objects[:, 0:batchsize] = matrices["object" + str(i)]
            Full[:, 0:batchsize] = matrices["full" + str(i)]

        downloadImgs(Background.real, Objects.real, Full.real, x_pix=x_pix, y_pix=y_pix, num=batchsize,
                     backpath='D:/background/', objpath='D:/objects/', fullpath='D:/output/', flag= i*batchsize)


def showImgs_batch_error(matrices, n, x_pix, y_pix):
    G = readgt(num=n, filepath="D:\groundtruth")
    num = int(len(matrices)/3)
    m = matrices["background0"].shape[0]
    batchsize = matrices["background0"].shape[1]
    Background = np.zeros((m,batchsize), dtype='complex')
    Objects = np.zeros((m,batchsize), dtype='complex')
    Full = np.zeros((m,batchsize), dtype='complex')
    Groundtruth = np.zeros((m,batchsize))
    start = 0
    end = batchsize
    errors = 0

    for i in range(num):
        if (i == num-1) and (np.mod(n,batchsize) != 0):
            Background[:, 0:] = matrices["background" + str(i)]
            Objects[:, 0:] = matrices["object" + str(i)]
            Full[:, 0:] = matrices["full" + str(i)]
            Groundtruth[:,0:] = Groundtruth[:,end:]

        else:
            Background[:, 0:batchsize]= matrices["background" + str(i)]
            Objects[:, 0:batchsize] = matrices["object" + str(i)]
            Full[:, 0:batchsize] = matrices["full" + str(i)]
            Groundtruth[:, 0:batchsize] = G[:,start:end]

        start = end
        end = end + batchsize

        downloadImgs(Background.real, Objects.real, Full.real, x_pix=x_pix, y_pix=y_pix, num=batchsize,
                     backpath='D:/background/', objpath='D:/objects/', fullpath='D:/output/', flag= i*batchsize)

        error, E = errorComputation(Objects.real, Groundtruth, x_pix, y_pix)
        errors = errors +error
        showimages(A=E, x_pix=x_pix, y_pix=y_pix, num= E.shape[1], filepath='D:/error/', flag=i * batchsize)
    return errors

def downloadImgs(background, objects, full, x_pix, y_pix, num, backpath, objpath, fullpath, flag):
    showimages(A=objects, x_pix=x_pix, y_pix=y_pix, num=num, filepath=objpath, flag=flag)
    showimages(A=background, x_pix=x_pix, y_pix=y_pix, num=num, filepath=backpath, flag=flag)
    showimages(A=full, x_pix=x_pix, y_pix=y_pix, num=num, filepath=fullpath, flag=flag)

def errorComputation(Objects, G, x_pix, y_pix):
    ImgNo = Objects.shape[1]
    Error = G - Objects
    error = np.sum(np.sum(Error))/ImgNo/x_pix/y_pix
    print("sub-error: " + str(error))
    return error, Error










