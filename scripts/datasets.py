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
        self.images     = []
        self.pixelInfo  = []
        self.other      = []
        self.vecs       = []
        self.vecLen     = None

        dc= listdir(saxFolder)[0]
        dic     = pdc.read_file(self.sFolder+'/'+dc)

        self.patient = self.dicToPatient(dic)  
        self.sliceLoc= dic.SliceLocation 
        self.approxL = int(float(self.sliceLoc))                
            


    def getData(self,imageProc):
        dcs             = sorted(listdir(self.sFolder),key=lambda x: self.dNameToNum(x))
        for dc in dcs:
            dic     = pdc.read_file(self.sFolder+'/'+dc)
            
            #deal with image data
            image   = imageProc.processImage(dic.pixel_array)
            self.images.append(image)
            shape   = dic.pixel_array.shape

            #deal with pixel info
            pixInfo = self.dicToPixInfo(dic,shape)            
            self.pixelInfo.append(pixInfo)

            #deal with other relevant data
            other   = self.dicToOther(dic)
            self.other.append(other)            
            #self.dicoms.append(dic)
            vec     = self.toVec(pixInfo,other)
            if len(self.vecs) == 0:
                self.vecLen     = vec.shape[0]
            self.vecs.append(vec)
            
    
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
      

    def toVec(self,pixInfo,other):
        out     = np.append(np.array(self.patient),pixInfo)
        out     = np.append(out,other)
        return out
        
       

        

#This is a set of short axis ShortSlices at various locations
class ShortStack():
    
    def __init__(self,saxList,imageProc):
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
        self.imageProc  = ImageProcessor()
        self.saxList    = sorted([self.pathname+'/'+fi for fi in self.folder if fi.find("sax") > -1])
        self.shortStack = ShortStack(self.saxList,self.imageProc)
        self.patient    = self.shortStack.patient
        self.ID         = self.pathToID()        
        
    def pathToID(self):
        p   = self.pathname.replace("/study",'')
        p   = p[p.rfind("/")+1:]
        return p
        
        
        
        
        
class ImageProcessor():
    
    def __init__(self,osize=(128,128),rotate=True,subsample=0.9):
        self.rotate     = rotate*np.random.rand()
        self.osize      = osize
        self.subsample  = subsample
        self.rg         = 10
        self.yx         = None
        self.angle      = None
        
        
    def processImage(self,image):
        #if this is the first time we see an image, decide a subsample and a rotation
        if self.yx  == None:
            self.yx     = self.fit(image)
            self.angle = np.random.uniform(-self.rg, self.rg)
            
        image   = self.randomShift(image)
        
        #if the image is wider than tall, flip it turnways
        if image.shape[1] > image.shape[0]:
            image   = ndimage.interpolation.rotate(image, 90, axes=(0,1),mode='nearest', reshape=True)

        #clip the highest intensity pixels             
        mu  = np.mean(image)
        sig = np.std(image)
        image[image > mu+2*sig] = mu+2*sig
        
        #global contrast normalization
        image   = (image*1.-np.mean(image)) / np.std(image)
         
        #rotate slightly for data augmentation purposes
        if self.rotate:
            image   = self.random_rotation(image,self.angle)
                 
        #resize the image to output size
        image   = self.resizeImage(image)
        return image
     
    def randomShift(self,image):
        return image[self.yx[0]:self.yx[0]+self.subsample*image.shape[0],self.yx[1]:self.yx[1]+self.subsample*image.shape[1]]
    
    def resizeImage(self,image):
        return mi.imresize(image,self.osize)
        
        
    def fit(self,image):
        y,x   = image.shape
        yshift= int(np.random.uniform(0,max(y-self.subsample*y,1)))
        xshift= int(np.random.uniform(0,max(x-self.subsample*x,1)))
        print image.shape, yshift, xshift
        return yshift, xshift


    def random_rotation(self,x, cval=0.):
        x = ndimage.interpolation.rotate(x, self.angle, axes=(0,1),mode='nearest', reshape=True, cval=cval)
        return x