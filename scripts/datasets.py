import dicom  as pdc
import numpy as np
from os import listdir
import scipy.misc as mi
from scipy import ndimage

#This is a set of dicom files associated with a location (slice) of the heart
class ShortAxisSlice():
    
    def __init__(self,saxFolder):
        self.sFolder    = saxFolder
        self.saxInd     = saxFolder[saxFolder.rfind("_")+1:]
        #self.dicoms     = []
        self.patient    = None
        self.images     = []
        self.pixelInfo  = []
        self.other      = []


        dcs             = sorted(listdir(saxFolder),key=lambda x: self.dNameToNum(x))
        for dc in dcs:
            dic     = pdc.read_file(self.sFolder+'/'+dc)
            if self.patient is None:
                self.patient = self.dicToPatient(dic)  
                self.sliceLoc= dic.SliceLocation 
                self.approxL = int(float(self.sliceLoc))                
            
            #deal with image data
            image   = self.processImage(dic.pixel_array)
            self.images.append(image)
            shape   = image.shape

            #deal with pixel info
            pixInfo = self.dicToPixInfo(dic,shape)            
            self.pixelInfo.append(pixInfo)

            #deal with other relevant data
            self.other.append(self.dicToOther(dic))            
            #self.dicoms.append(dic)
        
        #print self.sliceLoc
            
    
    #returns the index of a dicom file (at a time t)
    def dNameToNum(self,name):
        ind1    = name.rfind("-")
        name    = name[ind1+1:-4]
        return int(name)      
        
    def dicToPatient(self,dic):
        pat     = []
        pat.append(int(dic.PatientAge[:-1]))
        pat.append((dic.PatientSex=="M")*1.)
        return pat

    def dicToPixInfo(self,dic,shape):
        pixInfo     = []
        pixInfo.append(dic.LargestImagePixelValue)
        pixInfo.append(dic.SmallestImagePixelValue)
        pixInfo.append(dic.PixelSpacing[0])
        pixInfo.append(dic.PixelSpacing[1])
        pixInfo.append(dic.SamplesPerPixel)
        pixInfo.append(dic.PixelBandwidth)
        pixInfo.append(shape[0])
        pixInfo.append(shape[1])
        return pixInfo
        
    def dicToOther(self,dic):
        other   = []
        other.append(dic.SliceThickness)
        other.append(dic.SliceLocation)
        other.append(dic.RepetitionTime)
        return other
        
        
    def processImage(self,image):
        image   = image.astype(np.float32)
        image   -= np.mean(image)
        image   /= np.std(image)+0.000001
        
        return np.array(image)
        

        

#This is a set of short axis ShortSlices at various locations
class ShortStack():
    
    def __init__(self,saxList):
        self.stack   = []
        self.patient = None
        saxList      = sorted(saxList,key=lambda x: self.dNameToNum(x))
        for sax in saxList:
            self.stack.append(ShortAxisSlice(sax))
            if self.patient == None:
                self.patient    = self.stack[0].patient
                
        self.stack  = sorted(self.stack, key=lambda x: (x.approxL,x.saxInd))
        lastStackLoc= -1000000
        dupes       = []
        for num, s in enumerate(self.stack):
            if abs(s.approxL - lastStackLoc) <= 1:
                dupes.append(num-1)
            lastStackLoc    = s.approxL

        self.stack  = [val for ind,val in enumerate(self.stack) if ind not in dupes] 
        self.uniqueSlices   = len(self.stack)
        
        
    #returns the index of a sax slice from the folder name
    def dNameToNum(self,name):
        ind1    = name.rfind("_")
        name    = name[ind1+1:]
        return int(name)


class Study():
    
    def __init__(self,pathname):
        self.pathname   = pathname+'/study'
        self.folder     = listdir(self.pathname) 
        self.saxList    = sorted([self.pathname+'/'+fi for fi in self.folder if fi.find("sax") > -1])
        self.shortStack = ShortStack(self.saxList)
        self.patient    = self.shortStack.patient
        self.ID         = self.pathToID()        
        
    def pathToID(self):
        p   = self.pathname.replace("/study",'')
        p   = p[p.rfind("/")+1:]
        return p
        
        
        
        
        
class ImageProcessor():
    
    def __init__(self,osize=(64,64),rotate=True,subsample=(230,230)):
        self.rotate     = rotate*np.random.rand()
        self.osize      = osize
        self.subsample  = subsample
        self.yx         = None
        
        
    def getOutputImage(self,image):
        if self.yx  == None:
            self.yx     = self.fit(image)
        
        
    def fit(self,image):
        y,x   = image.shape
        yshift= max(self.subsample[0]-y,0)
        xshift= max(self.subsample[1]-x,0)
        return yshift, xshift


    def random_rotation(self,x, rg, fill_mode="nearest", cval=0.):
        angle = random.uniform(-rg, rg)
        x = ndimage.interpolation.rotate(x, angle, axes=(1,2), reshape=False, mode=fill_mode, cval=cval)
        return x