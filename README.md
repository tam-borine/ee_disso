**Note:** this repo is still under construction, code needs to be added and refactored and if you try to run things they probably won't work because dependencies might be missing, let alone be reproducible. Coming soon.

However an outline of the contents of this repo:

* A pre-processing pipeline which uses the Google Earth Engine (GEE) API to acquire images, create labelled data and export

* I make some difference images with the GEE API between pre and during flood images to see the change

* The ML input pipeline is for my machine learning model which ingests tfrecords from S1 images we exported from GEE

* The main model trains a CNN to do semantic segmentation of S1 images from GEE delineating flooded areas
