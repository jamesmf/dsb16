# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 09:39:48 2016

@author: test
"""

from os.path import isdir
from os import mkdir
from os.path import isfile
import numpy as np
import sys
import h5py


"""*******************************************************************"""            
"""                     Helper Functions                             """
"""*******************************************************************"""

"""if update == False, create a new folder (with 'run' being higher than highest in current outTyep folder)
   else just define the folder name"""
def defineFolder(update,outType,size,run):
    if run != '':
        run     = "_"+run
    
    folder  = "../"+outType+'/'+str(size)+run+"/"
        
    if (run == "_1") and (isdir(folder)) and (not update):
        i=1
        oldfolder = folder[:folder.rfind("_")+1]
        while isdir(folder):
            i+=1
            folder  = oldfolder[:-1]+"_"+str(i)+'/'
            print folder
            
    if not update:
        mkdir(folder)
        mkdir(folder+"tempTest/")        
        mkdir(folder+"tempTrain/")
    return folder


def handleArgs(arglist,size=200):  
    if len(arglist) < 2:
        print "needs folder as input"
        sys.exit(1)
    else:
        folder  = arglist[1]
        temp    = folder[folder.find("/")+1:]
        targetType    = temp[:temp.find("/")]
        if folder[-1] == "/":
            folder = folder[:-1]
        f1      = folder[folder.rfind("/")+1:]
        f2      = f1[:f1.find("_")]
        f3      = f1[f1.find("_")+1:]   
        size    = int(f2)                    #size of the images
        #print f3
        run     = f3
        


    print  "size: ", size, "run: ", run, "targetType: ", targetType
    return size, run, targetType

"""get the systole/diastole values for each study"""
def getTargets(ttv,targetType="both"):
    targ    = readCSV("../data/"+ttv+".csv")
    out     = {}
    if targetType   == "both":
        for row in targ:
            out[row[0]] = row[1:]
    elif targetType == "systole":
        for row in targ:
            out[row[0]] = row[1]
    elif targetType == "diastole":
        for row in targ:
            out[row[0]] = row[2]
    
    return out
    

"""Read a csv without csv"""   
def readCSV(fname):
    out     = []
    with open(fname,'rb') as f:
        t   = [x for x in f.read().split("\n") if x != '']
    out     = [x.split(",") for x in t]
    out     = np.array(out)
    return out
    

"""Given a dictionary of target values, return the length of the output"""    
def getOutSize(outputs):
    try:
        return len(outputs[outputs.keys()[0]])
    except TypeError:
        return 1
        
        
"""*******************************************************************"""            
"""                       Saving Loading                              """
"""*******************************************************************"""        
def saveModel(model,location):
    jsonstring  = model.to_json()
    with open(location+".json",'wb') as f:
        f.write(jsonstring)
    model.save_weights(location+"weights.h5",overwrite=True)
    
def loadModel(location):
    from keras.models import model_from_json
    with open(location+".json",'rb') as f:
        json_string     = f.read()
    model = model_from_json(json_string)
    model.load_weights(location+"weights.h5")
    return model
    
def saveData(data,location):
    h5f     = h5py.File(location+".h5",'w')
    h5f.create_dataset('data1',data=data)
    h5f.close()

            
def loadData(location):
    h5f     = h5py.File(location+".h5",'r')
    data    = h5f['data1'][:]
    h5f.close()
    return data
   
"""*******************************************************************"""            
"""                        Data Processing                            """
"""*******************************************************************"""

def getStack(study,n_slices):
    newStack    = []
    toPop      = []
    if len(study.shortStack.stack) < n_slices:
        print "study has too few slices"
        print study.ID
        return None
        
    while len(toPop) < n_slices:
        x   = int(np.floor(len(study.shortStack.stack)*np.random.rand()))
        if x not in toPop:
            toPop.append(x)
    toPop   = sorted(toPop)
    for n,tp in enumerate(toPop):
        newStack.append(study.shortStack.stack.pop(tp-n))
        #print n,tp,toPop, len(newStack), len(study.shortStack.stack)
    return newStack

def saxToImages(sl):
    pass
    
def saxToRaw(sl):
    return np.reshape(sl.vecs,(len(sl.vecs),sl.vecLen))
    
def studyToSingleTime(st,n_time=30,n_slices=5):
    stack   = getStack(st,n_slices)
    if stack is None:
        return None
        
    raws    = []
    ims     = []
    for sl in stack:
        sl.getData(st.imageProc)
        raws.append(saxToRaw(sl))
        ims.append(sl.images)
        
    raws    = np.reshape(raws,(len(raws)*raws[0].shape[0],raws[0].shape[1]))
    images  = np.reshape(ims,(len(ims)*len(ims[0]),ims[0][0].shape[0],ims[0][0].shape[1]))
    return raws, images