import os
import sys
import numpy as np
import time

def readRating(line):
    line = line.split(",")
    return [int(line[0]), int(line[1]),int(line[2])]

def genperms(numworkers):
    return np.random.permutation(numworkers)+1


def assignBlockIndex (index, numData, numWorkers):
    blockSize = numData/numWorkers
    if(numData % numWorkers != 0): blockSize = blockSize + 1
    return int(np.floor(index/np.ceil(blockSize)))+1

def main(numFactors, numWorkers, maxIter, beta, lam):
    #conf = SparkConf().setAppName('DSGD-MF').setMaster('local')
    #sc = SparkContext(conf=conf)
    numWorkers = int(numWorkers)
    maxIter = int(maxIter)
    numFactors = int(numFactors)
    beta = float(beta)
    lam = sc.broadcast(lam)
    
    
   
    #M = sc.textFile("/FileStore/tables/dtkgosw61464757508966/TrainingRatingsShuf.txt").map(readRating)
    M = sc.textFile("/FileStore/tables/3dymcg5b1464757763335/RatingsShuf.txt").map(readRating).persist()
    #M = sc.textFile("/FileStore/tables/9lkfxx6x1464574874711/ratings_tiny-993e4.txt").map(readRating)

    start = time.time()
    numRows = M.max(lambda x : x[0])[0] +1
    numCols = M.max(lambda x : x[1])[1] +1
    avgRating = M.map(lambda x: x[2]).mean()
    
    scaleRating = np.sqrt(avgRating * 4 / numFactors)
    print avgRating
    
    tau_0 = 100
    

    mseList = []
    times = []
    #mse = sc.accumulator(0)
    ## Main Loop
    Mblocked = M.keyBy(lambda x: assignBlockIndex(x[0], numRows, numWorkers)).partitionBy(numWorkers)
    #print Mblocked.collect()
   
    W = M.map(lambda x: tuple([int(x[0]),1])).reduceByKey(lambda x,y : x+y).map(lambda x: tuple([x[0],tuple([x[1],scaleRating*np.random.rand(1, numFactors).astype(np.float32)])])).persist()
    #print W.collect()
    H = M.map(lambda x: tuple([int(x[1]),1])).reduceByKey(lambda x,y : x+y).map(lambda x: tuple([x[0],tuple([x[1],scaleRating*np.random.rand(numFactors,1).astype(np.float32)])])).persist()
    for it in range(maxIter):
        mse = sc.accumulator(0.0)
        nUpdates = sc.accumulator(0.0)
        stepSize = sc.broadcast(np.power(tau_0 + it, -beta))
        #for s in range(numWorkers-1):
        perms = genperms(numWorkers)
        Mfilt = Mblocked.filter(lambda x: perms[x[0]-1]==assignBlockIndex(x[1][1],numCols,numWorkers)).persist()
        Hblocked = H.keyBy(lambda x: perms[assignBlockIndex(x[0], numRows, numWorkers)-1])
        Wblocked = W.keyBy(lambda x: assignBlockIndex(x[0], numRows, numWorkers))
        groupRDD = Mfilt.groupWith(Hblocked, Wblocked).partitionBy(numWorkers)
        Mfilt.unpersist()
        WH = groupRDD.mapPartitions(lambda x: SGD(x, stepSize, numFactors,lam, mse, nUpdates,scaleRating)).reduceByKey(lambda x,y: x+y).persist()

        W = WH.filter(lambda x: x[0]=='W').flatMap(lambda x: x[1]).persist()
        H = WH.filter(lambda x: x[0]=='H').flatMap(lambda x: x[1]).persist()
            #Wvec = W.first()
            #Hvec = H.first()
        Wvec = W.collect()
        Hvec = H.collect()
        mseCur = mse.value / nUpdates.value
        mseList.append(mseCur)
        times.append(time.time()-start)
        if mseCur < 0.65:
          break


  
    print time.time()-start
    print times
    print mseList
    



def SGD(keyed_iterable, stepSize, numFactors,lam, mse, nUpdates, scaleRating):
    iterlist = (keyed_iterable.next())
    Miter = iterlist[1][0]
    Hiter = iterlist[1][1]
    Witer = iterlist[1][2]
    
    Wdict = {}
    Hdict = {}
    
    Wout = {}
    Hout = {}
    
    for h in Hiter:
        Hdict[h[0]] = h[1]
    
    for w in Witer:
        Wdict[w[0]] = w[1]
    
    for m in Miter:
        (i,j,rat) = m
        
        if i in Wdict:
            W_input = Wdict[i]
        else:
            Wdict[i] = tuple([i,scaleRating*np.random.rand(1,numFactors).astype(np.float32)])
            W_input = Wdict[i]
        if j in Hdict:
            H_input = Hdict[j]
        else:
            Hdict[j] = tuple([j,scaleRating*np.random.rand(numFactors,1).astype(np.float32)])
            H_input = Hdict[j]

        (Nh, Hprev) = H_input
        (Nw, Wprev) = W_input
        delta = -2*(rat - Wprev.dot(Hprev))
        Wnew = Wprev - stepSize.value*(delta*Hprev.T + (2.0*lam.value/Nh)*Wprev)
        Hnew = Hprev - stepSize.value*(delta*Wprev.T + (2.0*lam.value/Nw)*Hprev)
        mse += (rat - Wnew.dot(Hnew))**2

        nUpdates += 1

        Wout[i] = tuple([Nw, Wnew])
        Hout[j] = tuple([Nh, Hnew])
        
    return (tuple(['W',Wout.items()]), tuple(['H',Hout.items()]))