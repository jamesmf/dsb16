import sys
import helper
import numpy as np
from os import listdir
from os.path import isfile
#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape,Merge, TimeDistributedDense
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta ,SGD, Adagrad, RMSprop
from keras.layers.recurrent import  GRU,LSTM
from keras.layers.extra import TimeDistributedFlatten, TimeDistributedConvolution2D, TimeDistributedMaxPooling2D
#sys.setrecursionlimit(10000)
import time
import subprocess
np.random.seed(0)




    


"""Define parameters of the run"""
size            = 128                        #EDIT ME!   #how large the images are
outType         = "crnn" 

#direct          = "../data/SDF/"            #directory containing the SD files
#ld              = listdir(direct)                   #contents of that directory
#shuffle(ld)                                 #shuffle the image list for randomness

numSlices       = 5
timeSteps       = 30*numSlices
batch_size      = 4                        #how many training examples per batch
chunkSize       = 50000                     #how much data to ever load at once      
testChunkSize   = 6000                      #how many examples to evaluate per iteration
mDataSize       = (timeSteps,13)
print mDataSize
run             = "1"



"""Define the folder where the model will be stored based on the input arguments"""
folder          = helper.defineFolder(False,outType,size,run)
print folder
trainDirect     = "../data/train/"
valDirect       = "../data/validate/"
#testDirect      = "../data/test/"
trainNP         = folder+"tempTrain/"

trainFs = listdir(trainDirect)
valFs   = listdir(valDirect)
#testFs  = listdir(testDirect)
trainL  = len(trainFs)
valL    = len(valFs)
#testL   = len(testFs)


print "training examples : ", trainL
print "validation examples : ", valL
#print "test examples : ", testL

features     = helper.getTargets("train")            #get the target vector for each CID
outsize             = helper.getOutSize(features)



"""DEFINE THE MODEL HERE"""  

imageModel = Sequential()
imageModel.add(TimeDistributedConvolution2D(8, 3, 3, border_mode='valid', input_shape=(timeSteps,1,size,size)))
imageModel.add(Activation('relu'))
imageModel.add(TimeDistributedConvolution2D(16, 3, 3, border_mode='valid'))
imageModel.add(TimeDistributedMaxPooling2D(pool_size=(2, 2),border_mode='valid'))
imageModel.add(Activation('relu'))
imageModel.add(TimeDistributedConvolution2D(32, 3, 3, border_mode='valid'))
imageModel.add(TimeDistributedMaxPooling2D(pool_size=(2, 2),border_mode='valid'))
imageModel.add(Activation('relu'))
imageModel.add(TimeDistributedConvolution2D(64, 3, 3, border_mode='valid'))
imageModel.add(Activation('relu'))
imageModel.add(TimeDistributedFlatten())



flatModel   = Sequential()
flatModel.add(LSTM(output_dim=32,return_sequences=True,input_shape=mDataSize))


model   = Sequential()
model.add(Merge([imageModel, flatModel],mode='concat'))
model.add(LSTM(output_dim=512,return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(output_dim=256,return_sequences=False))
model.add(Dense(256))
model.add(Dropout(.2))
model.add(Dense(outsize))


lr  = 0.00001
optimizer   = Adadelta()
model.compile(loss='mean_squared_error', optimizer=optimizer)


helper.saveModel(model,folder+"wholeModel")

""" TRAINING """
superEpochs     = 1000
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
    print trainImages.shape, trainVecs.shape, trainTargets.shape
    
   
    model.fit([trainImages, trainVecs],trainTargets, batch_size=batch_size,nb_epoch=1)
#    
#    for b in range(0,trainImages.shape[0]/batch_size):
#        tI  = trainImages[b*batch_size:(b+1)*batch_size]
#        tV  = trainVecs[b*batch_size:(b+1)*batch_size]
#        tT  = trainTargets[b*batch_size:(b+1)*batch_size]
#        model.train_on_batch([tI,tV], tT)

   
    del trainImages, trainTargets


