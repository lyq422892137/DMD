# inputs: X, Y, rank
# outputs:
# operators: phi, b, w, t，M(Koopman operator)

from svd import rsvd
from numpy import *

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

    return phi, B, V

def compute_Mhat(U,Y,V,S):
    M_hat = dot(U.T, Y).dot(V).dot(S)
    return M_hat

def compute_eig(X):
    # a, b = np.linalg.eig(x)
    # a is eigenvalues, b is eigenvector
    # here， L = a, W = b
    L, W = linalg.eig(X)
    return L, W

def compute_phi(Y,V,S,W):
    phi = dot(Y, V).dot(S).dot(W)
    return phi

def compute_B(phi,rank_new, col):
    b = dot(phi.T,phi).I.dot(phi.T).dot(col)
    B = mat(eye(rank_new) * array(b))
    return B, b

# def geneV(rank_new, n, L):
#     V = ones((rank_new, n), dtype=complex)
#     for i in range(len(L)):
#         V[i, :] = L[i]
#         for t in range(n):
#             V[i, t] = V[i, t] ** t
#     return V

def geneV(rank_new, n, L):
    V = ones((rank_new, n), dtype=complex)
    fmode = log(L)
    for i in range(len(L)):
        V[i, :] = fmode[i]
        for t in range(n):
            V[i, t] = V[i, t] * t
    V = exp(V)
    return V

def geneV_fmode(rank_new, n, L, threshold = 0.001):

    V1 = ones((rank_new, n), dtype=complex)
    V2 = ones((rank_new, n), dtype=complex)
    V3 = ones((rank_new, n), dtype=complex)
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
            # V2[j, :] = 255

    # print("V1")
    # print(V1)
    # print("V2")
    # print(V2)

    return V1, V2, V3


def compute_newD(phi,B,V):
    D_new = dot(phi, B).dot(V)
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

    return phi, B, V1, V2, V3

def rDMD_batch(X, Y, D, rank=5, p=5, q=5, threshold = 0.001, batchsize = 100):
    n = D.shape[1]
    if n <= batchsize:
        object_extraction(X, Y, D, rank, p, q, threshold)
    else:
        M = {}
        parameters = {}
        output = {}
        Dstart = 0
        Dend = batchsize
        subStart = 0
        subEnd = batchsize - 1

        rank_new = int(n / batchsize)

        for i in range(int(n / batchsize)):
            M["D" + str(i)] = D[:, Dstart:Dend]
            M["X" + str(i)] = X[:, subStart:subEnd]
            M["Y" + str(i)] = Y[:, subStart:subEnd]

            parameters['phi' + str(i)], parameters['B' + str(i)], parameters['V1' + str(i)], parameters[
                'V2' + str(i)], parameters['V3' + str(i)] = object_extraction(X=M["X" + str(i)], Y=M["X" + str(i)],
                                                                              D=M["D" + str(i)],
                                                                              rank=rank_new, p=p, q=q,
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
            M["D" + str(rank_new)] = D[:, Dstart:]
            M["X" + str(rank_new)] = X[:, subStart:]
            M["Y" + str(rank_new)] = Y[:, subStart:]

            parameters['phi' + str(rank_new)], parameters['B' + str(rank_new)], parameters['V1' + str(rank_new)], parameters[
                'V2' + str(rank_new)], parameters['V3' + str(rank_new)] = object_extraction(X=M["X" + str(rank_new)], Y=M["X" + str(rank_new)],
                                                                              D=M["D" + str(rank_new)],
                                                                              rank= int(mod(n,batchsize)/batchsize * rank_new), p=p, q=q,
                                                                              threshold=threshold)

            output["background" + str(rank_new)] = compute_newD(parameters['phi' + str(rank_new)], parameters['B' + str(rank_new)],
                                                         parameters['V1' + str(rank_new)])
            output["object" + str(rank_new)] = compute_newD(parameters['phi' + str(rank_new)], parameters['B' + str(rank_new)],
                                                     parameters['V2' + str(rank_new)])
            output["full" + str(rank_new)] = compute_newD(parameters['phi' + str(rank_new)], parameters['B' + str(rank_new)],
                                                   parameters['V3' + str(rank_new)])

        return output, parameters