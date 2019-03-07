import numpy as np
from PIL import Image
import cv2
import json
from pycocotools.coco import COCO as co
datas=co('RCTW_train.json')
visnum=1
ims=datas.getImgIds()
for i in range(visnum):
    imn=datas.loadImgs(ims[i])[0]['file_name']
    im=np.array(Image.open('images/'+imn).convert('RGB'))
    im=im[...,::-1]
    im_copy=im.copy()
    anns=datas.loadAnns(datas.getAnnIds(ims[i]))
    for ann in anns:
        bbox=ann['bbox']
        bbox[2]+=bbox[0]
        bbox[3]+=bbox[1]
        cv2.rectangle(im_copy,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0,0),2)
    cv2.imwrite('vis/'+imn,im_copy)
