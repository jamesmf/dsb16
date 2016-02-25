import datasets 
from os import listdir
import numpy as np

fold    = "../data/train/"
ld      = sorted(listdir(fold),key=lambda x : int(x))
studies = [fold+s for s in ld]

lengths     = []
for s in studies:
    study   = datasets.Study(s)
    stack   =  study.shortStack.stack
    print study.shortStack.uniqueSlices
    lengths.append(study.shortStack.uniqueSlices)
print np.sum(lengths), np.mean(lengths), np.max(lengths), np.min(lengths)