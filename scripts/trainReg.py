import datasets 
from sklearn import linear_model
import numpy as np
from os import listdir
from scipy.stats import norm

def readCSV(fname):
    out     = []
    with open(fname,'rb') as f:
        t   = [x for x in f.read().split("\n") if x != ''][1:]
    out     = [x.split(",") for x in t]
    out     = np.array(out)
    return out


fold    = "../data/train/"
trainTs = readCSV("../data/train.csv")
ld      = sorted(listdir(fold),key=lambda x : int(x))
studies = [fold+s for s in ld]

lengths     = []
Xtrain      = np.zeros((len(studies),2))
ytrain      = np.zeros((len(studies),2))
for n,s in enumerate(studies):
    study   = datasets.Study(s)
    Xtrain[n,:] = np.array(study.patient)
    ytrain[n]   = np.array(trainTs[n,1:])
    
LR  = linear_model.Ridge(alpha=0.5)
LR.fit(Xtrain,ytrain)


fold    = "../data/validate/"
ld      = sorted(listdir(fold),key=lambda x : int(x))
studies = [fold+s for s in ld]

lengths     = []
Xtest       = np.zeros((len(studies),2))
IDs         = []
for n,s in enumerate(studies):
    study       = datasets.Study(s)
    Xtest[n,:]  = np.array(study.patient)
    IDs.append(study.ID)
    
preds   =  LR.predict(Xtest)
std     = 10.

systoles    = {}
diastoles   = {}
for n,ID in enumerate(IDs):
    s   = []
    d   = []
    systole     = preds[n,0]
    diastole    = preds[n,1]
    for i in range(0,600):
        s_i     = norm.ppf(systole-(i*1.)/std)
        s.append(s_i)
        d_i     = norm.ppf(diastole-(i*1.)/std)
        d.append(s_i)
    systoles[ID]    = s
    diastoles[ID]   = d

with open("../submission.csv",'wb') as f:
    f.write("ID")
    [f.write(",P"+str(i)) for i in range(0,600)]
    f.write("\n")

    for ID in IDs:
        f.write(str(ID)+"_Diastole,")
        f.write(','.join(diastoles[ID]))
        f.write("\n"+str(ID)+"_Systole,")
        f.write(','.join(systoles[ID]))
        f.write("\n")
