import datasets 
import helper
from os import listdir
import numpy as np
import scipy.misc as mi
import sys
from os.path import isfile
import time
import helper

fold    = "../data/train/"
ld      = sorted(listdir(fold),key=lambda x : int(x))
studies = [fold+s for s in ld]

chunkSize   = 20

lengths     = []
Ximages     = []
Xvecs       = []
ys          = []
studyInd    = 0

trainTargets    = helper.getTargets("train")

mu  = None
sig = None

while True:
    while len(Ximages) < chunkSize:
        
        s                   = studies[studyInd]
        study               = datasets.Study(s)  
        sid                 = study.ID
        data                = helper.studyToSingleTime(study)
        if not data is None:
            Xvecs.append(data[0])
            Ximages.append(data[1])
            ys.append(trainTargets[sid])
            
        studyInd +=1
        if studyInd == len(studies):
            studyInd    = 0
            np.random.shuffle(studies)

    XtrainImage     = np.reshape(Ximages,(len(Ximages),len(Ximages[0]),1,Ximages[0][0].shape[0],Ximages[0][0].shape[1]))
    XtrainVecs      = np.reshape(Xvecs,(len(Xvecs),len(Xvecs[0]),len(Xvecs[0][0])))
    ytrain          = np.reshape(ys,(len(ys),2))
    
    if mu is None:
        tempX   = np.reshape(XtrainVecs,(XtrainVecs.shape[0]*XtrainVecs.shape[1],XtrainVecs.shape[2]))
        mu      = np.mean(tempX,axis=0)
        sig     = np.std(tempX,axis=0)+0.00000001

    XtrainVecs  = XtrainVecs - mu
    XtrainVecs  = XtrainVecs/sig

    while isfile(sys.argv[1]+"tempTrain/XtrainImages.h5"):   
        print "sleeping because Train folder full             \r",
        time.sleep(1.)    
    
    helper.saveData(XtrainImage,sys.argv[1]+"tempTrain/XtrainImages")
    helper.saveData(XtrainVecs,sys.argv[1]+"tempTrain/XtrainVecs")
    helper.saveData(ytrain,sys.argv[1]+"tempTrain/ytrain")
    
    #mi.imsave("../images/inmatrix.jpg",XtrainImage[0,0,0,:,:])
    
    print XtrainImage.shape
    print XtrainVecs.shape    
    
    Xvecs       = []
    Ximages     = []
    ys          = []
    
