# cal_f_measure

this is the source code from [icdar website](http://rrc.cvc.uab.es)

but it seems that the download link fails to work

## Install

some python module is needed

- shapely (no pip version for windows)
- matplotlib
- tqdm (can be removed)
- glob (os.walk have replace it)
- pickle (a standard module for py3 but for python2 need Cpickle)
- shutil
- csv

## Prepare Data

ground truth files should be placed in **./gt**, which should have the name of image_*.txt. The data in file should have the format of (x1,y1,x2,y2,x3,y3,x4,y4)

result files should be placed in **./train**, we have the same filename of gt files.The data in file should have the format of (x1,y1,x2,y2,x3,y3,x4,y4,scores)

## Running

python fmeasure.py

the result will be in **./save**
