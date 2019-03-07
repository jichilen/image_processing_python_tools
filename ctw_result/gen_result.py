import os
import numpy as np
import argparse
import json
import pycocotools.mask as mask_util
from skimage import measure
parser = argparse.ArgumentParser(description="convert segentation results for coco-style mask")
parser.add_argument("--data_dir",type=str,required=True)
args = parser.parse_args()
rootdir=args.data_dir
rname='segm.json'
results=json.load(open(os.path.join(rootdir,rname)))
thresh=0.9
# {'category_id': 1,
#  'image_id': 1001,
#  'score': 0.9890751838684082,
#  'segmentation': {'counts': '******',
#   'size': [240, 180]}}
cal=0
for re in results:
	if re['score']<thresh:
		continue
	# cal+=1
	# if cal>2:
	# 	break
	seg=re['segmentation']
	seg=mask_util.decode(seg)
	contours=measure.find_contours(seg,0.5)
	with open('result/'+str(re['image_id'])+'.txt','a') as f:
		for contour in contours:
			contour=contour[:,[1,0]]
			line=list(contour.reshape(-1))
			line=[str(l) for l in line]
			line=','.join(line)
			f.write(line+'\n')



