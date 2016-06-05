# -*- coding: utf-8 -*-
#!/usr/bin/python


from operator import mul
from itertools import imap, izip, starmap
import numpy as np
from numba import typeof, double, int_
from numba.decorators import autojit, jit
import shelve
import time
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

#time for 5000 iterations = 47 mins
#MSE: 0.579

@autojit(locals={'step': int_, 'e': double, 'err': double}) 
def sgdmf(R, W, H, K, inds,start_time):
    steps = 10
    alpha = 0.001
    lam = 0.1
    errors = []
    times = []
    for step in xrange(steps):
        print step
        mse = 0.0
        i1 = 0
        for ind in inds:
            i = ind[0]
            j = ind[1]
          
            eij = R[i1]
            for p in xrange(K):
                eij -= W[i,p] * H[j,p]

            for k in xrange(K):
                W[i,k] += alpha * (2 * eij * H[j,k] - lam * W[i,k])
                H[j,k] += alpha * (2 * eij * W[i,k] - lam * H[j,k])

            nR = np.dot(W[i,:],H[j,:])
        
            mse += (R[i1] - nR)**2
            i1 = i1+1
        err = mse / len(inds)
        errors.append(err)
        print err
        times.append(time.time()-start_time)
        if err < 0.65:
            break

    return W, H, step, err, errors, times


def main(hist=False, seed=1234567, num_factors=50):
   # start_time = time.time()    
    X = np.loadtxt('ml-1m/RatingsShuf.txt', delimiter = ',', usecols = (0,1,2))
    shape = tuple(X.max(axis=0)[:2]+1)
    R = coo_matrix((X[:,2],(X[:,0],X[:,1])),shape = shape, dtype = X.dtype)
    inds = zip(R.row, R.col)
    R = R.data
    print(len(R))

    N, M = [int(shape[0]), int(shape[1])]
    K = num_factors
    W = np.random.rand(N, K)
    H = np.random.rand(M, K)
    start_time = time.time()
    nW, nH, steps, error, errors, times = sgdmf(R, W, H, K, inds,start_time)
    endtimefit = time.time()
    print('Convergence in %i steps' % steps)

    plt.plot(np.arange(steps+1),errors)
    plt.show()
    mse = 0
    i1 = 0
    for ind in inds:
        i = ind[0]
        j = ind[1]
      
        eij = R[i1]
        for p in xrange(K):
            eij -= nW[i,p] * nH[j,p]
        mse += eij**2

    print mse/len(inds)

    print('Train time in minutes: %.4f' \
           % ((endtimefit-start_time) / 60.0))

    print('Factorization MSE: %.3f' % error)
    f=open('f50.txt','w')
    for i in range(len(times)):
        f.write(str(times[i])+','+str(errors[i])+'\n')

    f.close() 


if __name__ == "__main__":
    main()