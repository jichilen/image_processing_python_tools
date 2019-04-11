'''
preprocess tools
class boxlist is used
usage:
    if self.extra_aug is not None:
        img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes,
                                                   gt_labels)
'''
import mmcv
import numpy as np
from numpy import random
from pycocotools.coco import COCO
import cv2
from PIL import Image
from shapely.geometry import Polygon

def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
            bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


class PhotoMetricDistortion(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img, boxes, segments, labels):
        '''
        get image distortion results when call the object

        :param img:
        :param boxes:
        :param labels:
        :return:
        '''
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        return img, boxes, segments, labels


class Expand(object):

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, boxes, segments, labels):
        if random.randint(2):
            return img, boxes, segments, labels

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        boxes += np.tile((left, top), 2)
        out_segments = []
        for segment in segments:
            segment = np.array(segment).reshape(-1, 2)
            segment += (left, top)
            out_segments.append(list(segment.reshape(-1)))
        return img, boxes, out_segments, labels


class RandomCrop(object):

    def __init__(self,
                 min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, boxes, segments, labels):
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return img, boxes, segments, labels

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array((int(left), int(top), int(left + new_w),
                                  int(top + new_h)))
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                        center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                               center[:, 1] < patch[3])
                if not mask.any():
                    continue
                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                # box here is not valid enough
                # we shound use mask to generate the new box
                # boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                # boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                # boxes -= np.tile(patch[:2], 2)
                boxes=[]
                out_segments = []
                out_labels=[]
                for j,segment in enumerate(segments):
                    if not mask[j]:
                        continue
                    segment = np.array(segment).reshape(-1, 2)
                    segment=Polygon(segment)
                    bound=patch.copy()
                    bound=bound+np.array([1,1,-1,-1])
                    bound=np.vstack((bound[[0,2,2,0]],bound[[1,1,3,3]])).transpose()
                    bound=Polygon(bound)
                    segment = bound.intersection(segment)
                    try:
                        segment = np.array(segment.exterior.coords)
                    except Exception as e:
                        print(e)
                        continue

                    segment -= patch[:2]
                    x1,y1=np.min(segment,0)
                    x2,y2=np.max(segment,0)
                    boxes.append([x1,y1,x2,y2])
                    out_labels.append(labels[j])
                    out_segments.append(list(segment.reshape(-1)))
                boxes=np.array(boxes).astype(np.int32)
                return img, boxes, out_segments, np.array(out_labels)


class ExtraAugmentation(object):

    def __init__(self,
                 photo_metric_distortion=None,
                 expand=None,
                 random_crop=None):
        self.transforms = []
        if photo_metric_distortion is not None:
            self.transforms.append(
                PhotoMetricDistortion(**photo_metric_distortion))
        if expand is not None:
            self.transforms.append(Expand(**expand))
        if random_crop is not None:
            self.transforms.append(RandomCrop(**random_crop))

    def __call__(self, img, boxes, segments, labels):
        img = img.astype(np.float32)
        for transform in self.transforms:
            img, boxes, segments, labels = transform(img, boxes, segments, labels)
        return img, boxes, segments, labels


if __name__ == '__main__':
    filename = '/data2/data/ctw1500/ctw1500_test.json'
    imgp = '/data2/data/ctw1500/test/text_image/'
    res = COCO(filename)
    imgids = res.getImgIds()
    distort = {
        'brightness_delta': 32,
        'contrast_range': (0.5, 1.5),
        'saturation_range': (0.5, 1.5),
        'hue_delta': 18,
    }
    expand = {
        'mean': (0, 0, 0),
        'to_rgb': True,
        'ratio_range': (1, 4)
    }
    randcrop = {
        'min_ious': (0, 0.1, 0.3),#(0.1, 0.3, 0.5, 0.7, 0.9)
        'min_crop_size': 0.3
    }
    auth = ExtraAugmentation(distort, expand, randcrop)
    for i in range(20):
        imn = imgp + res.loadImgs(imgids[i])[0]['file_name']
        im = np.array(Image.open(imn).convert('RGB'))
        im = im[..., ::-1]  # convert to GBR
        annids = res.getAnnIds(imgids[i])
        anns = res.loadAnns(annids)
        boxes = []
        segs = []
        labels = []
        for ann in anns:
            labels.append(ann['category_id'])
            boxes.append(ann['bbox'])
            segs.append(ann['segmentation'])
        boxes = np.array(boxes)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        # segs = np.array(segs)  # segs is still a list for they may have different lenth
        labels = np.array(labels)
        imout = im.copy()
        for box, seg in zip(boxes, segs):
            seg = np.array(seg).reshape((-1, 2))
            cv2.rectangle(imout, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
            cv2.polylines(imout, [seg], True, (0, 0, 255), 1)
        cv2.imwrite('a{}_0.jpg'.format(i), imout)


        im, boxes, segs, labels = auth(im, boxes, segs, labels)
        im_o = im.copy()
        print(im_o.shape)
        for box, seg in zip(boxes, segs):
            seg = np.array(seg).reshape((-1, 2)).astype(np.int32)
            print(seg)
            cv2.rectangle(im_o, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv2.polylines(im_o, [seg], True, (0, 0, 255), 2)
        cv2.imwrite('a{}.jpg'.format(i), im_o)
