from svd import rsvd
from numpy import *
import gc
import time

def rdmd(X, Y, D, rank=5, p=5, q=5):
    # compute X's svd:
    random.seed(7)
    rank_new = rank + p
    Ux, sigmax, Vx = rsvd(X, rank, p, q)
    Sx = mat(diag(sigmax)).I

    # compute M_hat
    M_hat = compute_Mhat(Ux, Y, Vx, Sx)

    # compute L and W
    # L is eigenvalues, W is eigenvector
    L, W = compute_eig(M_hat)

    # compute phi (dynamic modes)  and B, the amplitudes
    phi = compute_phi(Y, Vx, Sx, W)

    B, b = compute_B(phi, rank_new, X[:,0])

    V = geneV(rank_new, D.shape[1], L)

    del M_hat, L, W, X, Y, D, Ux, Sx, Vx, b
    gc.collect()

    return phi, B, V

def compute_Mhat(U,Y,V,S):
    M_hat = dot(U.T, Y).dot(V).dot(S)
    del U, Y, V, S
    gc.collect()
    return M_hat

def compute_eig(X):
    # a, b = np.linalg.eig(x)
    # a is eigenvalues, b is eigenvector
    # here， L = a, W = b
    L, W = linalg.eig(X)
    del X
    gc.collect()
    return L, W

def compute_phi(Y,V,S,W):
    phi = dot(Y, V).dot(S).dot(W)
    del Y, V, S, W
    gc.collect()
    return phi

def compute_B(phi,rank_new, col):
    b = dot(phi.T,phi).I.dot(phi.T).dot(col)
    B = mat(eye(rank_new) * array(b))
    del phi, rank_new, col
    gc.collect()
    return B, b

def geneV(rank_new, n, L):
    V = zeros((rank_new, n), dtype=complex)
    fmode = log(L)
    for i in range(len(L)):
        V[i, :] = fmode[i]
        for t in range(n):
            V[i, t] = V[i, t] * t
    V = exp(V)
    del rank_new, n, L, fmode
    gc.collect()
    return V

def geneV_fmode(rank_new, n, L, threshold):
    V1 = zeros((rank_new, n), dtype=complex)
    V2 = zeros((rank_new, n), dtype=complex)
    V3 = zeros((rank_new, n), dtype=complex)
    fmode = log(L)

    for i in range(len(L)):
        V1[i, :] = fmode[i]
        V2[i, :] = fmode[i]
        V3[i, :] = fmode[i]

        for t in range(n):
            V1[i, t] = V1[i, t] * t
            V2[i, t] = V2[i, t] * t
            V3[i, t] = V3[i, t] * t

    V1 = exp(V1)
    V2 = exp(V2)
    V3 = exp(V3)

    for j in range(len(L)):
        if abs(fmode[j]) <= threshold:
            V2[j,:] = 0
        else:
            V1[j,:] = 0

    # print("V1")
    # print(V1)
    # print("V2")
    # print(V2)
    # print("V3")
    # print(V3)
    del fmode, L, rank_new, threshold
    gc.collect()

    return V1, V2, V3


def compute_newD(phi,B,V):
    D_new = dot(phi, B).dot(V)
    del phi, B, V
    gc.collect()
    return D_new

def object_extraction(X, Y, D, rank, p, q, threshold):
    random.seed(7)
    rank_new = rank + p
    Ux, sigmax, Vx = rsvd(X, rank, p, q)
    Sx = mat(diag(sigmax)).I
    M_hat = compute_Mhat(Ux, Y, Vx, Sx)
    L, W = compute_eig(M_hat)

    # compute phi (dynamic modes)  and B, the amplitudes
    phi = compute_phi(Y, Vx, Sx, W)

    B, b = compute_B(phi, rank_new, X[:, 0])

    V1, V2, V3 = geneV_fmode(rank_new, D.shape[1], L, threshold=threshold)

    del X, Y, D, rank, p, q, threshold
    gc.collect()

    return phi, B, V1, V2, V3

def rDMD_batch(X, Y, D, rank=5, p=5, q=5, threshold = 0.001, batchsize = 100):
    n = D.shape[1]
    M = {}
    parameters = {}
    output = {}

    if n <= batchsize:
        parameters["phi0"], parameters["B0"], parameters["V10"], parameters["V20"], parameters["V30"] \
            = object_extraction(X=X, Y=Y, D=D, rank=rank, p=p, q=q, threshold=threshold)
        output["background0"] = compute_newD(parameters['phi0'], parameters['B0'], parameters['V10'])
        output["object0"] = compute_newD(parameters['phi0'], parameters['B0'], parameters['V20'])
        output["full0"] = compute_newD(parameters['phi0'], parameters['B0'], parameters['V30'])

    else:
        Dstart = 0
        Dend = batchsize
        subStart = 0
        subEnd = batchsize - 1

        rank_new = int((rank + p) * batchsize/n)
        num = int(n / batchsize)

        for i in range(num):
            print("round " + str(i) + ":")

            M["D" + str(i)] = D[:, Dstart:Dend]
            M["X" + str(i)] = X[:, subStart:subEnd]
            M["Y" + str(i)] = Y[:, subStart:subEnd]

            parameters['phi' + str(i)], parameters['B' + str(i)], parameters['V1' + str(i)], parameters[
                'V2' + str(i)], parameters['V3' + str(i)] = object_extraction(X=M["X" + str(i)], Y=M["Y" + str(i)],
                                                                              D=M["D" + str(i)],
                                                                              rank=rank_new, p=0, q=q,
                                                                              threshold=threshold)

            output["background" + str(i)] = compute_newD(parameters['phi' + str(i)], parameters['B' + str(i)],
                                                         parameters['V1' + str(i)])
            output["object" + str(i)] = compute_newD(parameters['phi' + str(i)], parameters['B' + str(i)],
                                                     parameters['V2' + str(i)])
            output["full" + str(i)] = compute_newD(parameters['phi' + str(i)], parameters['B' + str(i)],
                                                   parameters['V3' + str(i)])

            Dstart = Dend
            Dend = Dend + batchsize
            subStart = subEnd
            subEnd = subEnd + batchsize


        if mod(n, batchsize) != 0:
            M["D" + str(num)] = D[:, Dstart:]
            M["X" + str(num)] = X[:, subStart:]
            M["Y" + str(num)] = Y[:, subStart:]

            parameters['phi' + str(num)], parameters['B' + str(num)], parameters['V1' + str(num)], parameters[
                'V2' + str(num)], parameters['V3' + str(num)] = object_extraction(X=M["X" + str(num)], Y=M["Y" + str(num)],
                                                                              D=M["D" + str(num)],
                                                                              rank= int(rank_new * mod(n,batchsize) / batchsize), p=0, q=q,
                                                                              threshold=threshold)

            output["background" + str(num)] = compute_newD(parameters['phi' + str(num)], parameters['B' + str(num)],
                                                         parameters['V1' + str(num)])
            output["object" + str(num)] = compute_newD(parameters['phi' + str(num)], parameters['B' + str(num)],
                                                     parameters['V2' + str(num)])
            output["full" + str(num)] = compute_newD(parameters['phi' + str(num)], parameters['B' + str(num)],
                                                   parameters['V3' + str(num)])

    del Dstart, Dend, subEnd, subStart, X, Y, D
    gc.collect()
    return output, parameters

def errorComputation(Objects, G, x_pix, y_pix):
    ImgNo = Objects.shape[1]
    Error = G - Objects
    error = np.sum(np.sum(Error))/ImgNo/x_pix/y_pix
    print("sub-error: " + str(error))
    return error, Error




