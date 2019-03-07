import numpy as np
import os
from PIL import Image
import json
import cv2
print("please work on py3 environment")
import pickle
import tqdm
import random

def checkl(s,height,width):
    #s none 2
    for i in range(s.shape[0]):
        s[i,0]=min(width-1,s[i,0])
        s[i,1]=min(height-1,s[i,1])
        s[i,0]=max(0,s[i,0])
        s[i,1]=max(0,s[i,1])
    return s

curdir=os.getcwd()
train_imgdir=        #'train_images/'
train_txtdir=        #'train_labels.json'
json_t_out=          #'art_train.json'
json_v_out=          #'art_val.json'
data_t={}
data_v={}
categories=[{u'id': 1, u'name': u'text', u'supercategory': u'text'}]
annotations=[]
images=[]
#annotations u'area': 1114.0,u'bbox': [45, 56, 91, 29],u'category_id': 1,u'id': 7692,u'image_id': 1001,u'iscrowd': 0,u'segmentation':[[x,y,x,y]]
#images  u'file_name': u'1001.jpg', u'height': 240, u'id': 1001, u'width': 180

#get imname of dataset
imnames=os.listdir(train_imgdir)

#need to split train and val dataset
n_im=len(imnames)
random.shuffle(imnames)
with open(train_txtdir)as f:
    gts = json.load(f)
iid=0
anid=0
flag=0
for ni,imn in enumerate(tqdm.tqdm(imnames,ascii=True)):
    # load information for each image
    image={}
    iid+=1
    im=Image.open(os.path.join(train_imgdir,imn))
    image[u'file_name']=imn
    image[u'height']=im.height
    image[u'width']=im.width
    image[u'id']=iid
    images.append(image)

    # res is used for anns in ones images
    res=
    for re in res:
        # check whether data is ignored

        # init
        annotation={}
        anid+=1

        #segmentation should in form of n*2 usually n=4
        segmentation=
        segmentation=np.array(segmentation)
        # check the validation of annotation
        segmentation=checkl(segmentation,im.height,im.width)
        x1=int(np.min(segmentation[:,0]))
        y1=int(np.min(segmentation[:,1]))
        x2=int(np.max(segmentation[:,0]))
        y2=int(np.max(segmentation[:,1]))
        area=int((x2-x1)*(y2-y1))
        bbox=[x1,y1,x2-x1,y2-y1]
        annotation['area']= area
        annotation['bbox'] = bbox
        annotation['category_id'] = 1
        annotation['id'] = anid
        annotation['image_id'] = iid
        annotation['iscrowd'] = 0
        annotation['segmentation'] = [segmentation.reshape((-1)).tolist()]
        annotations.append(annotation)
    if ni > 0.9*n_im and flag==0:
        data_t['images']=images
        data_t['categories']=categories
        data_t['annotations']=annotations
        images=[]
        annotations=[]
        flag=1
data_v['images']= images
data_v['categories']=categories
data_v['annotations']=annotations
with open(json_v_out,'w',ensure_ascii=False,indent=4)as f:
    json.dump(data_v,f)
with open(json_t_out,'w',ensure_ascii=False,indent=4)as f:
    json.dump(data_t,f)

