import os
import numpy as np
import argparse
import json
# import pycocotools.mask as mask_util
from skimage import measure
from voc_eval_polygon import voc_eval_polygon

parser = argparse.ArgumentParser(description = "convert segentation results for coco-style mask")
parser.add_argument("--data_dir", type = str, required = True)
args = parser.parse_args()
rootdir = args.data_dir
rname = 'segm.json'
results = json.load(open(os.path.join(rootdir, rname)))
thresh = 0.9
# {'category_id': 1,
#  'image_id': 1001,
#  'score': 0.9890751838684082,
#  'segmentation': {'counts': '******',
#   'size': [240, 180]}}
cal = 0
for re in results:
    if re['score'] < thresh:
        continue
        # cal+=1
        # if cal>2:
        # 	break
    seg = re['segmentation']
    seg = mask_util.decode(seg)
    contours = measure.find_contours(seg, 0.5)
    with open('result/' + str(re['image_id']) + '.txt', 'a') as f:
        for contour in contours:
            contour = contour[:, [1, 0]]
            line = list(contour.reshape(-1))
            line = [str(l) for l in line]
            line = ','.join(line)
            f.write(line + '\n')

anno_path = 'result/'
outputstr = 'result_out/text'
# score_thresh_list=[0.2, 0.3, 0.4, 0.5, 0.6, 0.62, 0.65, 0.7, 0.75, 0.8, 0.9]
score_thresh_list = [0.5]
files = os.listdir(anno_path)
files.sort()
for iscore in score_thresh_list:
    with open(outputstr + str(iscore) + '.txt', "w") as f1:
        for ix, filename in enumerate(files):
            # print(filename)
            imagename = filename[:-4]
            # print(imagename)

            with open(os.path.join(anno_path, filename), "r") as f:
                lines = f.readlines()

            for line in lines:
                box = line.strip().split(",")
                assert (len(box) % 2 == 0), 'mismatch xy'
                out_str = "{} {}".format(str(int(imagename[:]) - 1001), 0.999)
                for i in box:
                    out_str = out_str + ' ' + str(i)
                f1.writelines(out_str + '\n')

detpath = 'result_out/{:s}.txt'

annopath = '/data2/data/ctw1500/test/test_label_curve.txt'
imagesetfile = '/data2/data/ctw1500/test/test.txt'

score_thresh_list = [0.5]
for isocre in score_thresh_list:
    rec, prec, ap = voc_eval_polygon(detpath[:-4] + str(isocre) + '.txt', annopath, imagesetfile, 'text',
                                     ovthresh = 0.5)

file = './txt_pr.txt'
_ = lambda x, y: 2 * x * y * 1.0 / (x + y)
with open(file, 'w') as f:
    f.write('ap     rec    prec   f-measure\n')
    for i in range(len(rec)):
        f.write('{:.4f} {:.4f} {:.4f} {:.4f}\n'.format(ap, rec[i], prec[i], _(rec[i], prec[i])))
    print('ap: {:.4f}, recall: {:.4f}, pred: {:.4f}, FM: {:.4f}\n'.format(ap, rec[-1], prec[-1], _(rec[-1], prec[-1])))
