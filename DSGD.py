import os
import sys
import multiprocessing
import csv
import numpy as np
from numpy import linalg
from scipy import sparse
from operator import itemgetter
import time
from pyspark import SparkContext, SparkConf

def readRating(line):
    line = line.split(",")
    return [int(line[0]), int(line[1]), float(line[2])]

def genperms(numworkers):
    return np.random.permutation(numworkers)+1


def assignBlockIndex (index, numData, numWorkers):
    blockSize = numData/numWorkers
    if(numData % numWorkers != 0): blockSize = blockSize + 1
    return int(np.floor(index/np.ceil(blockSize)))+1

def main(numFactors, numWorkers, maxIter, beta, lam):
    conf = SparkConf().setAppName('DSGD-MF').setMaster('local')
    sc = SparkContext(conf=conf)
    numWorkers = int(numWorkers)
    maxIter = int(maxIter)
    numFactors = int(numFactors)
    beta = float(beta)
    lam = sc.broadcast(lam)
    
    
   
    M = sc.textFile("ml-1m/RatingsShuf.txt").map(readRating)
    start = time.time() 
    numRows = M.max(lambda x : x[0])[0] +1
    numCols = M.max(lambda x : x[1])[1] +1

    
    tau_0 = 100
    

    mseList = []
    ## Main Loop
    Mblocked = M.keyBy(lambda x: assignBlockIndex(x[0], numRows, numWorkers)).partitionBy(numWorkers)
    #print Mblocked.collect()
    ini_divide = pow(1.0/numFactors,0.5)

    W = M.map(lambda x: tuple([int(x[0]),1])).reduceByKey(lambda x,y : x+y).map(lambda x: tuple([x[0],tuple([x[1],ini_divide*np.random.rand(1, numFactors).astype(np.float32)])]))
    #print W.collect()
    H = M.map(lambda x: tuple([int(x[1]),1])).reduceByKey(lambda x,y : x+y).map(lambda x: tuple([x[0],tuple([x[1],ini_divide*np.random.rand(numFactors,1).astype(np.float32)])]))
    for it in range(maxIter):
        mse = sc.accumulator(0)
        stepSize = sc.broadcast(np.power(tau_0 + it, -beta))
        for s in range(numWorkers-1):
            perms = genperms(numWorkers)
            Mfilt = Mblocked.filter(lambda x: perms[x[0]-1]==assignBlockIndex(x[1][1],numCols,numWorkers))
            Hblocked = H.keyBy(lambda x: perms[assignBlockIndex(x[0], numRows, numWorkers)-1]).partitionBy(numWorkers)
            Wblocked = W.keyBy(lambda x: assignBlockIndex(x[0], numRows, numWorkers)).partitionBy(numWorkers)
            groupRDD = Mfilt.groupWith(Hblocked, Wblocked)
            WH = groupRDD.mapPartitions(lambda x: SGD(x, lam, stepSize, numFactors,mse)).reduceByKey(lambda x,y: x+y)

            W = WH.filter(lambda x: x[0]=='W').flatMap(lambda x: x[1])
            H = WH.filter(lambda x: x[0]=='H').flatMap(lambda x: x[1])  
            Wvec = W.collect()
            Hvec = H.collect()          

        mseList.append(mse.value)
        if mse.value < 0.5: 
            break



    print mseList
    print time.time() - start



def SGD(keyed_iterable, lam, stepSize, numFactors, mse):
    iterlist = (keyed_iterable.next())
    Miter = iterlist[1][0]
    Hiter = iterlist[1][1]
    Witer = iterlist[1][2]
    
    Wdict = {}
    Hdict = {}
    
    Wout = {}
    Hout = {}
    ini_divide = pow(1.0/numFactors,0.5)
    for h in Hiter:
        Hdict[h[0]] = h[1]
    
    for w in Witer:
        Wdict[w[0]] = w[1]
    length = len(Miter)
    for m in Miter:
        (i,j,rat) = m
        
        if i in Wdict:
            W_input = Wdict[i]
        else:
            Wdict[i] = tuple([i,ini_divide*np.random.rand(1,numFactors).astype(np.float32)])
            W_input = Wdict[i]
        if j in Hdict:
            H_input = Hdict[j]
        else:
            Hdict[j] = tuple([j,ini_divide*np.random.rand(numFactors,1).astype(np.float32)])
            H_input = Hdict[j]

        (Nh, Hprev) = H_input
        (Nw, Wprev) = W_input
        delta = -2*(rat - Wprev.dot(Hprev))
        Wnew = Wprev - stepSize.value*(delta*Hprev.T + (2.0*lam.value/Nh)*Wprev)
        Hnew = Hprev - stepSize.value*(delta*Wprev.T + (2.0*lam.value/Nw)*Hprev)
        mse += (rat - Wnew.dot(Hnew))**2  / float(length)
        


        Wout[i] = tuple([Nw, Wnew])
        Hout[j] = tuple([Nh, Hnew])

    
        
    return (tuple(['W',Wout.items()]), tuple(['H',Hout.items()]))

        



if __name__ == "__main__":
    main(50, 4, 5, 0.8, 0.1)