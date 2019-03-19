import numpy as np
from pycocotools.coco import COCO as coco
import os,sys
import pickle
from shapely.geometry import *

class Masker(object):
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """

    def __init__(self, threshold=0.5, padding=1):
        self.threshold = threshold
        self.padding = padding

    def forward_single_image(self, masks, boxes):
        boxes = boxes.convert("xyxy")
        im_w, im_h = boxes.size
        res = [
            paste_mask_in_image(mask[0], box, im_h, im_w, self.threshold, self.padding)
            for mask, box in zip(masks, boxes.bbox)
        ]
        if len(res) > 0:
            res = torch.stack(res, dim=0)[:, None]
        else:
            res = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))
        return res

    def __call__(self, masks, boxes):
        if isinstance(boxes, BoxList):
            boxes = [boxes]

        # Make some sanity check
        assert len(boxes) == len(masks), "Masks and boxes should have the same length."

        # TODO:  Is this JIT compatible?
        # If not we should make it compatible.
        results = []
        for mask, box in zip(masks, boxes):
            assert mask.shape[0] == len(box), "Number of objects should be the same."
            result = self.forward_single_image(mask, box)
            results.append(result)
        return results

def curve_parse_rec_txt(filename):
    with open(filename.strip(),'r') as f:
        gts = f.readlines()
        objects = []
        for obj in gts:
            cors = obj.strip().split(',')
            cors = [int(i) for i in cors]
            obj_struct = {}
            obj_struct['name'] = 'text'
            obj_struct['difficult'] = 0
            lup=np.array(cors[:2]).reshape(-1,2)
            poly=np.array(cors[4:]).reshape(-1,2)
            poly=poly+lup
            obj_struct['bbox'] = list(poly.reshape(-1))
            objects.append(obj_struct)
    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval_polygon(detpath,
             annpath,
             classname,
             ovthresh=0.5,
             use_07_metric=False):

    # first load gt
    assert ('json' in annpath)," only for data in cocoformat"
    cachefile = annpath.split('/')[-1][:-4]+'pkl'#json
    # read list of images
    gts=coco(annpath)
    ims=gts.getImgIds()
    ims.sort()
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imid in enumerate(ims):
            anids=gts.getAnnIds(imid)
            anns=gts.loadAnns(anids)
            objs=[]
            for ann in anns:
                obj={}
                seg=ann['segmentation']
                if isinstance(seg[0],list):
                    assert len(seg)==1,"sth wrong in json file"
                    seg=seg[0]
                obj['name'] = 'text'
                obj['difficult'] = 0
                obj['bbox']=seg
                objs.append(obj)
            print(imid,' loaded')
            recs[i] = objs
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                                    i + 1, len(ims)))
        # save
        print ('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    class_recs = {}
    npos = 0
    for ix in range(len(recs.keys())):
        R = [obj for obj in recs[ix] if obj['name'] == classname] # text 
        # assert(R), 'Can not find any object in '+ classname+' class.'
        if not R: continue
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[str(ims[ix])] = {'bbox': bbox,
                                'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    # BB = BB[sorted_ind, :]
    # image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d] # mask rcnn
        det_bbox = bb[:]
        pts = [(det_bbox[j], det_bbox[j+1]) for j in range(0,len(bb),2)]
        try:
            pdet = Polygon(pts)
        except Exception as e:
            print(e)
            continue
        if not pdet.is_valid: 
            print('predicted polygon has intersecting sides.')
            # print(pts, image_ids[d])
            continue

        ovmax = -np.inf
        BBGT = R['bbox']
        # gt_bbox = BBGT[:, :4]
        info_bbox_gt = BBGT
        ls_pgt = [] 
        overlaps = np.zeros(BBGT.shape[0])
        for iix in range(BBGT.shape[0]):
            # box_gt = info_bbox_gt[iix]
            # pts=[(box_gt[j],box)]
            pts = [(info_bbox_gt[iix][j],  info_bbox_gt[iix][j+1]) for j in range(0,len(info_bbox_gt[iix]),2)]
            pgt = Polygon(pts)
            if not pgt.is_valid: 
                print('GT polygon has intersecting sides.')
                continue
            try:
                sec = pdet.intersection(pgt)
            except Exception as e:
                print('intersect invalid',e)
                continue
            try:
                assert(sec.is_valid), 'polygon has intersection sides.' # for mask rcnn
            except Exception as e:
                print(e)
                continue
            inters = sec.area
            uni = pgt.area + pdet.area - inters
            if uni <= 0.00001: uni = 0.00001
            overlaps[iix] = inters*1.0 / uni
            
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.
        print(R)
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return rec, prec, ap