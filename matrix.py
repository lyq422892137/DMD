from numpy import *
from svd import cal_svd
from svd import rsvd
from svd import svd_newMatrix
from svd import rsvd_newMatrix
import time

random.seed(7)

m = 3
n = 4
rank = 2
p = 0
B = random.random((m, n))
print(B)
A = random.random((m,n))
print(A)
print(sum(sum(A-B)))


time1= []
error1=[]

# for i in range(0,2):
#     begin1 = time.clock()
#     U, sigma, V = cal_svd(A,rank)
#     end1 = time.clock()
#     print("svd time: " + str(end1-begin1))
#     A_new, D = svd_newMatrix(A, U, sigma, V, rank)
#     time1.append(end1-begin1)
#     error1.append(D)


# print(A)
# print("-----------------------")
# print(A_new.shape)


# rsvd

time2= []
error2=[]

for i in range(0,10):
    begin2 = time.clock()
    U2, sigma2, V2 = rsvd(A=A, rank=rank, p=p)
    end2 = time.clock()
    print("rsvd time: " + str(end2-begin2))
    A_new2, D2 = rsvd_newMatrix(A, U2, sigma2, V2, rank, p = p)
    error2.append(D2)
    time2.append(end2-begin2)


print("-----------------------")
print("svd time mean:"+str(mean(time1)))
print("svd error mean:"+str(mean(error1)))
print("rsvd time mean:"+str(mean(time2)))
print("rsvd error mean:"+str(mean(error2)))
# print(A)
# print("-----------------------")
# print(A_new2)



