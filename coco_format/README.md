
### VOC数据生成

#### VOC数据格式

> +VOCdevkit
>
>     +VOC2012
>
>       +JPEGImages
>
>       +SegmentationClass
>
>       +ImageSets
>
>         +Segmentation

一个voc的数据目录如图所示，SegmentationClass里面存放的是分割结果，里面的图片都是**8-bit伪彩色图**，也就是说正常的读取我们读出来的是**24-bit RGB图**，但是其实这个图片只是一个8-bit的图片，进行了一次色彩的映射所以看起来给人一种彩色图的感觉

```python
import cv2
import numpy as np
from PIL import Image
#错误的读取方式
#im=cv2.imread("2011_003271.png")#w,h,c
#im=cv2.imread("2011_003271.png",0)#w,h
#正确的读取方式
im=np.array(Image.open('2011_003271.png'))#w,h
这样图里面存放的是正确的标签
```

那么问题来了这种图片是怎样生成的呢？

tensorflow官方贴出了一个做法，

```python
def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((_DATASET_MAX_ENTRIES[_PASCAL], 3), dtype=int)
  ind = np.arange(_DATASET_MAX_ENTRIES[_PASCAL], dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= bit_get(ind, channel) << shift
    ind >>= 3

  return colormap

####
colormap = create_label_colormap(_PASCAL)
colormap[label]
```

可以看到，先是构建了一个8bit - 24bit的一个映射，然后直接查表获得结果，但是这样做的话最后得到的结果依旧是一个24bit图

------

一个正确的做法如下图所示：

```
from utils import get_dataset_colormap
colormap=get_dataset_colormap.create_pascal_label_colormap()
palette=list(colormap.reshape(-1))
#im should have the mode L
im.putpalette(palette)  
```

这样可以保证生成的图片就是8位伪彩色图，但是需要注意的是，**生成的伪彩色图并不能直接用于训练**，因为绝大多数读图过程是不能区分伪彩色图和彩色图的，处于保险起见还是先转换出灰度图较为合适。但是伪彩色图能够在方便可视化的时候保留正确的gt值，这一点是我们所需要的。

### COCO数据格式

coco存储数据用的是一个字典字典里面有三个键：`"images", "categories", "annotations"`

里面装着一个list，list里面都是字典

具体的格式如下图所示

>images:
>
>	categories:
>
>		[
>
>		id:
>
>		name:
>
>		supercategory:
>
>		]
>
>	images:
>
>		[
>
>		file_name:
>
>		height:
>
>		width:
>
>		id:
>
>		]
>
>	annotations:
>
>		[
>
>		area:
>		bbox: #[x1,y1,w,h]
>		category_id:
>		id:
>		image_id:
>		iscrowd:
>   segmentation: #[[x,y,x,y]]
>
>		]


```python
 import numpy as np
 import os
 from PIL import Image
 import json
 from hanzi_util import is_zhs
 import cv2
 print("please work on py3 environment")
 import pickle
 import tqdm
 valdict=pickle.load(open('news_sohusite_xml.dat_JIEBA_PK','rb'))
 
 def checkl(s,height,width):
     #s none 2
     for i in range(s.shape[0]):
         s[i,0]=min(width-1,s[i,0])
         s[i,1]=min(height-1,s[i,1])
         s[i,0]=max(0,s[i,0])
         s[i,1]=max(0,s[i,1])
     return s
 
 curdir=os.getcwd()
 datadir=os.path.join(curdir,'icpr')
 train_imgdir=os.path.join(datadir,'image_9000')
 train_txtdir=os.path.join(datadir,'txt_9000')
 test_imgdir=os.path.join(datadir,'train_1000/image_1000')
 test_txtdir=os.path.join(datadir,'train_1000/txt_1000')
 data={}
 categories=[{u'id': 1, u'name': u'text', u'supercategory': u'text'}]
 annotations=[]
 images=[]
 #annotations u'area': 1114.0,u'bbox': [45, 56, 91, 29],u'category_id': 1,u'id': 7692,u'image_id': 1001,u'iscrowd': 0,u'segmentation':[[1,2,3]]
 #images  u'file_name': u'1001.jpg', u'height': 240, u'id': 1001, u'width': 180
 imnames=os.listdir(train_imgdir)
 iid=0
 anid=0
 for imn in tqdm.tqdm(imnames):
     image={}
     iid+=1
     if iid>300:
         break
     im=Image.open(os.path.join(train_imgdir,imn))
     image[u'file_name']=imn
     image[u'height']=im.height
     image[u'width']=im.width
     image[u'id']=iid
     images.append(image)
     with open(os.path.join(train_txtdir,imn[:-3]+'txt'),encoding='utf-8') as f:
         res=f.readlines()
     for re in res:
         re=re.split(',')
         # check whether it is chinese character
         valzhs=re[-1].strip()
         if '#' in valzhs:
             continue
         if not is_zhs(valzhs):
             valnum=0
         else:
             valnum=0
             for t in range(len(valzhs)-1):
                 valzh=valzhs[t:t+2]
                 if valzh in valdict and valdict[valzh]>0.01:
                     valnum+=1
         annotation={}
         anid+=1
         segmentation=[]
         for k in range(8):
             segmentation.append(int(float(re[k])))
         segmentation=np.array(segmentation)
         segmentation=segmentation.reshape((-1,2))
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
         annotation['valnum']=valnum
         annotations.append(annotation)
 data['images']= images
 data['categories']=categories
 data['annotations']=annotations
 with open('icpr_train_300.json','w')as f:
     json.dump(data,f)

```

