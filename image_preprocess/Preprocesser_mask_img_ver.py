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

    def __call__(self, img, boxes, gt_masks, gt_segs, labels):
        """
        apply transform when calling
        :param img: numpy.ndarray,(h,w,c)
        :param boxes: numpy.ndarray,(n,4), x1,y1,x2,y2
        :param segments: list(list),(n)(x),x is variant
        :param labels: numpy.ndarray,(n)
        :return:
        """

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

        return img, boxes, gt_masks, gt_segs, labels


class Expand(object):

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 2)):
        '''

        :param mean: background color for expand, defaul in gbr order
        :param to_rgb: whether to use rgb order
        :param ratio_range: scale fractor for expanding
        '''
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def expand_mask(self, ma, h, w, ratio, t, l):
        expand_img = np.full((int(h * ratio), int(w * ratio)), 0).astype(ma.dtype)
        expand_img[t:t + h, l:l + w] = ma
        return expand_img

    def __call__(self, img, boxes, gt_masks, gt_segs, labels):
        '''
        apply transform when calling
        :param img: numpy.ndarray,(h,w,c)
        :param boxes: numpy.ndarray,(n,4), x1,y1,x2,y2
        :param segments: list(list),(n)(x),x is variant
        :param labels: numpy.ndarray,(n)
        :return:
        '''
        if random.randint(2):
            return img, boxes, gt_masks, gt_segs, labels

        ratio = random.uniform(self.min_ratio, self.max_ratio)
        h, w, c = img.shape
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))

        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        # TODO: change mask seg
        # list mask and list seg
        gt_masks = [self.expand_mask(mask, h, w, ratio, top, left) for mask in gt_masks]
        gt_segs = [self.expand_mask(seg, h, w, ratio, top, left) for seg in gt_segs]
        boxes += np.tile((left, top), 2)

        return img, boxes, gt_masks, gt_segs, labels


class RandomCrop(object):

    def __init__(self,
                 min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size=0.3):
        '''
        # TODO: some special change should do to change this iou?
        :param min_ious: iou threshold for crop-img and gt-box
        :param min_crop_size: minimum cropsize for croping
        '''
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, boxes, gt_masks, gt_segs, labels):
        '''
        apply transform when calling
        :param img: numpy.ndarray,(h,w,c)
        :param boxes: numpy.ndarray,(n,4), x1,y1,x2,y2
        :param segments: list(list),(n)(x),x is variant
        :param labels: numpy.ndarray,(n)
        :return:
        '''
        h, w, c = img.shape
        print(img.shape)
        print([i.shape for i in gt_masks])
        print([i.shape for i in gt_segs])
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return img, boxes, gt_masks, gt_segs, labels

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
                out_masks = []
                boxes = []
                out_labels = []
                gt_segs = gt_segs[0]
                gt_segs = gt_segs[patch[1]:patch[3], patch[0]:patch[2]]
                for j, ma in enumerate(gt_masks):
                    ma = ma[patch[1]:patch[3], patch[0]:patch[2]]
                    if not mask[j]:
                        gt_segs[ma > 0] = 0
                        continue
                    y_l, x_l = np.where(ma > 0)
                    x1, x2 = np.min(x_l), np.max(x_l)
                    y1, y2 = np.min(y_l), np.max(y_l)
                    boxes.append([x1, y1, x2, y2])
                    out_masks.append(ma)
                    out_labels.append(labels[j])
                boxes = np.array(boxes).astype(np.int32)
                out_labels = np.array(out_labels)
                gt_segs = [gt_segs]
                return img, boxes, out_masks, gt_segs, out_labels


class Scale:
    # TODO:Maybe randomn scale is need? or be replaced by padding with scale
    pass


class RandomRotate:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.sample_mode = angles

    def __call__(self, img, boxes, gt_masks, gt_segs, labels):
        '''
        apply transform when calling
        :param img: numpy.ndarray,(h,w,c)
        :param boxes: numpy.ndarray,(n,4), x1,y1,x2,y2
        :param segments: list(list),(n)(x),x is variant
        :param labels: numpy.ndarray,(n)
        :return:
        '''
        mode = random.choice(self.sample_mode)
        if mode == 0:
            return img, boxes, gt_masks, gt_segs, labels
        # TODO: all of the images should be included in the rotated image
        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -mode, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # perform the actual rotation and return the image
        img = cv2.warpAffine(img, M, (nW, nH))
        out_masks = []
        boxes = []
        gt_segs = [cv2.warpAffine(seg, M, (nW, nH)) for seg in gt_segs]
        for ma in gt_masks:
            ma = cv2.warpAffine(ma, M, (nW, nH))
            y_l, x_l = np.where(ma > 0)
            x1, x2 = np.min(x_l), np.max(x_l)
            y1, y2 = np.min(y_l), np.max(y_l)
            boxes.append([x1, y1, x2, y2])
            out_masks.append(ma)
        boxes = np.array(boxes).astype(np.int32)
        return img, boxes, out_masks, gt_segs, labels


class ExtraAugmentation_new(object):

    def __init__(self,
                 photo_metric_distortion=None,
                 random_rotate=None,
                 expand=None,
                 random_crop=None):
        '''
        a class to perform image augmentation, including rotate, distortion, expand and crop
        :param random_rotate: dict() store parameters for rotate
        :param photo_metric_distortion: dict() store parameters for distortion
        :param expand: dict() store parameters for expand
        :param random_crop: dict() store parameters for crop
        '''
        self.transforms = []
        if photo_metric_distortion is not None:
            self.transforms.append(
                PhotoMetricDistortion(**photo_metric_distortion))
        if random_rotate is not None:
            self.transforms.append(RandomRotate(**random_rotate))
        if expand is not None:
            self.transforms.append(Expand(**expand))
        if random_crop is not None:
            self.transforms.append(RandomCrop(**random_crop))

    def __call__(self, img, boxes, gt_masks, gt_segs, labels):
        '''
        apply transform when calling
        :param img: numpy.ndarray,(h,w,c)
        :param boxes: numpy.ndarray,(n,4), x1,y1,x2,y2
        :param segments: list(list),(n)(x),x is variant
        :param labels: numpy.ndarray,(n)
        :return:
        '''
        img = img.astype(np.float32)
        for transform in self.transforms:
            img, boxes, gt_masks, gt_segs, labels = transform(img, boxes, gt_masks, gt_segs, labels)
        return img, boxes, gt_masks, gt_segs, labels


if __name__ == '__main__':
    filename = '/data2/data/ART/art_val.json'
    imgp = '/data2/data/ART/train_images/'
    res = COCO(filename)
    imgids = res.getImgIds()
    distort = {
        'brightness_delta': 32,
        'contrast_range': (0.5, 1.5),
        'saturation_range': (0.5, 1.5),
        'hue_delta': 18,
    }
    randrot = {
        'angles': [0, 45, 90, 135, 180],
    }
    expand = {
        'mean': (0, 0, 0),
        'to_rgb': True,
        'ratio_range': (1, 2)
    }
    randcrop = {
        'min_ious': (0, 0.1, 0.3),  # (0.1, 0.3, 0.5, 0.7, 0.9)
        'min_crop_size': 0.3
    }
    auth = ExtraAugmentation_new(distort, randrot, expand, randcrop)  #
    for i in range(10):
        imn = imgp + res.loadImgs(imgids[i])[0]['file_name']
        im = np.array(Image.open(imn).convert('RGB'))
        im = im[..., ::-1]  # convert to GBR
        annids = res.getAnnIds(imgids[i])
        anns = res.loadAnns(annids)
        boxes = []
        masks = []
        labels = []
        for ann in anns:
            labels.append(ann['category_id'])
            boxes.append(ann['bbox'])
            masks.append(res.annToMask(ann))
        boxes = np.array(boxes)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        # segs = np.array(segs)  # segs is still a list for they may have different lenth
        labels = np.array(labels)
        imout = im.copy()
        for box, ma in zip(boxes, masks):
            cv2.rectangle(imout, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
            imout[ma > 0] = imout[ma > 0] * 0.5 + ma[ma > 0, None] * np.array([255, 0, 0]) * 0.5
        cv2.imwrite('a{}_0.jpg'.format(i), imout.astype(np.uint8))
        segs = [sum(masks)]
        cv2.imwrite('a{}_0_s.jpg'.format(i), segs[0] * 255)
        im, boxes, masks, segs, labels = auth(im, boxes, masks, segs, labels)
        im_o = im.copy()
        print(im_o.shape)
        for box, ma in zip(boxes, masks):
            cv2.rectangle(im_o, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            im_o[ma > 0] = im_o[ma > 0] * 0.5 + ma[ma > 0, None] * np.array([255, 0, 0]) * 0.5
        cv2.imwrite('a{}_1.jpg'.format(i), im_o.astype(np.uint8))
        cv2.imwrite('a{}_1_s.jpg'.format(i), segs[0] * 255)
