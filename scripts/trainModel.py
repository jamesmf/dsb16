# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:58:44 2015

@author: test
"""


import helper
import numpy as np
from os.path import isfile
import cPickle
import sys
import subprocess
import time
from sklearn.metrics import mean_squared_error


def dumpWeights(model):     
    layercount  = 0
    for layer in model.layers:
        try:
            weights     = model.layers[layercount].get_weights()[0]
            size        = len(weights)
            if size < 100:
                with open(folder+"layer"+str(layercount)+".pickle",'wb') as f:
                    cp = cPickle.Pickler(f)
                    cp.dump(weights)
            else:
                pass
                
        except IndexError:
            pass
        layercount  +=1





"""Require an argument specifying whether this is an update or a new model, parse input"""
size, run, outType     = helper.handleArgs(sys.argv)


"""Define parameters of the run"""
DUMP_WEIGHTS    = True                     #will we dump the weights of conv layers for visualization
batch_size      = 4                        #how many training examples per batch


"""Define the folder where the model will be stored based on the input arguments"""
folder          = helper.defineFolder(True,outType,size,run)
print folder
trainNP         = folder+"tempTrain/"
print trainNP




model   = helper.loadModel(folder+"wholeModel")


""" TRAINING """
superEpochs     = 10000
RMSE            = 1000000
oldRMSE         = 1000000
for sup in range(0,superEpochs):
     
    oldRMSE     = min(oldRMSE,RMSE)
    print "*"*80
    print "TRUE EPOCH ", sup
    print "*"*80    

    count   = 0
    added   = 0

    #Wait for the other processes to dump a pickle file
    while not isfile(trainNP+"XtrainImages.h5"):   
        print "sleeping because Train folder empty             \r",
        time.sleep(1.)
    print ""

    
    #Load the training data   
    print "Loading np  training arrays"
    loadedUp    = False
    while not loadedUp:
        try:
           
            trainImages  = helper.loadData(trainNP+"XtrainImages")
            trainVecs    = helper.loadData(trainNP+"XtrainVecs")
            trainTargets = helper.loadData(trainNP+"ytrain")            
            
            loadedUp    = True
        except Exception as e:
            err     = e
            print err, "                              \r",
            time.sleep(2)
    print ""

    subprocess.call("rm "+trainNP+"XtrainImages.h5",shell=True)
    subprocess.call("rm "+trainNP+"XtrainVecs.h5",shell=True)
    subprocess.call("rm "+trainNP+"ytrain.h5",shell=True)

    #train the model on it
    print trainImages.shape, trainVecs.shape
    model.fit([trainImages,trainVecs], trainTargets, batch_size=batch_size, nb_epoch=1)

   
    del trainImages, trainTargets

    helper.saveModel(model,folder+"wholeModel")



