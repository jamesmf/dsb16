# dsb16
Train a Recurrent Convolutional Neural Network on the cardiac images from the 2016 Data Science Bowl

## Requirements
 - Theano
 - Keras
 - keras-extra
 
## Methodology
This approach takes some number of short axis slices (sax) for each study and lines up their dicom images in order (such that sax_i_image_30 is before sax_i+1_image_1). A recurrent convolutional neural  network sees each image, updates its internal representation, then outputs a guess for end systolic/diastolic volume
 
## How to run
The training data folder should be at ../data/train relative to the scripts folder. The script `createTrainingData.py` creates "Study" objects and outputs numpy arrays for the script `trainModel.py ` to train on.
