# inputs: X, Y, rank
# outputs:
# operators: phi, b, w, t，M(Koopman operator)

from svd import rsvd
from numpy import *

def rdmd(X, Y, D, rank=5,p=5,q=5):
    # compute X's svd:
    random.seed(7)
    rank_new = rank+p
    Ux, sigmax, Vx = rsvd(X,rank,p,q)
    Sx = mat(eye(rank_new) * sigmax).I

    # compute M_hat
    M_hat = dot(Ux.T,Y).dot(Vx).dot(Sx)

    # compute upper lambda L
    # a, b = np.linalg.eig(x)
    # a is eigenvalues, b is eigenvector
    # here， L = a, W = b
    L, W = linalg.eig(M_hat)
    # compute omega = ln(lambda)/delta t
    # for a standard video, delta t = 1
    # thus, omega = ln(L)
    omega = log(L)
    # print("W:"+ str(W)) W no problem
    # print("Vx:" + str(Vx)) Vx no problem, all are different
    print("Y:"+str(Y))
    # print("Sx:" + str(Sx)) no problem but the values are very big
    # compute phi and B, the amplitudes
    phi = dot(Y,Vx).dot(Sx).dot(W)
    print("phi:" + str((phi[0,0]==phi[0,0].all())))
    print(phi)
    print(phi.shape)
    # b = dot(phi.T,phi).I.dot(phi.T).dot(X[:,0])
    b = linalg.lstsq(phi,X[:,0])[0]
    print("b:"+str(b)) # different bs
    B = mat(eye(rank_new) * array(b))

    V = vander(L)
    print("V:" + str((V[:, 0] == V[:, 0]).all()))
    print(V.shape)
    print(V)
    print("omega[0]:" + str(omega[0]))
    print("V:" + str(V))


    return phi, B, omega, V

def compute_V(n,k,omega):
    V = ones((n, k), dtype=complex)
    for i in range(len(omega)):
        V[:, i] = omega[i]
    return V

# def computeImags(omega, phi, B, n, threshold = 1):
#     k = len(omega)
#     m = B.shape[0]
#     V = compute_V(n, k, omega)
#     l = []
#     s = []
#     l_count = []
#     s_count = []
#
#     for i in range(len(omega)):
#         if sqrt(power(omega[i],2)) < threshold:
#             l.append(omega[i])
#             l_count.append(i)
#         else:
#             s.append(omega[i])
#             s_count.append(i)
#

    # l = []
    # l_count = []
    # s = []
    # s_count = []
    # V = compute_V(n, k, omega, -1)
    #
    # for i in range(len(omega)):
    #     if power(omega[i],2) < threshold:
    #         l.append(B[i]*phi[i]*exp(V[]))
    #         l_count.append(i)
    #     else:
    #         s.append(omega[i])
    #         s_count.append(i)
    #
    print(len(omega))
    print(len(s))
    print(s_count)
    #
    # return l, s, l_count, s_count

def compute_newD(phi,B,n,k,omega,V):
    V2 = ones((k, n), dtype=complex)
    for i in range(len(omega)):
        V2[:, i] = omega[i]
    for t in range(n):
        V2[:,t] = V2[:,t] * t
    V2 = exp(V2)
    print(V2.shape)
    print("phi:" +str(phi.shape))
    # D_new = dot(phi, B).dot(V)
    D_new = dot(phi, B).dot(V2)
    print("-----------------------")
    print("D_new:" +str(D_new))
    print("-----------------------")
    return D_new









