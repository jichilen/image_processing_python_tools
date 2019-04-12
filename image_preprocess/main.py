import cv2
from Preprocesser import ExtraAugmentation
import numpy as np

def draw_img(imname,imout,boxes,segs):
    for box, seg in zip(boxes, segs):
        seg = np.array(seg).reshape((-1, 2)).astype(np.int32)
        cv2.rectangle(imout, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
        cv2.polylines(imout, [seg], True, (0, 0, 255), 1)
    cv2.imwrite(imname, imout)

def main():
    # some basic dicts for Augmentation
    randrot = {
        'angles': [0, 45, 90, 135, 180],
    }
    distort = {
        'brightness_delta': 32,
        'contrast_range': (0.5, 1.5),
        'saturation_range': (0.5, 1.5),
        'hue_delta': 18,
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
    # init the preprocesser
    auth = ExtraAugmentation(randrot, distort, expand, randcrop)  #
    # basic information for dataset
    im=cv2.imread('/data2/data/ART/train_images/gt_5454.jpg')
    boxes=np.array(
       [[ 77, 249, 306, 375],
        [394, 236, 456, 264],
        [503, 195, 645, 319],
        [ 72, 454, 303, 486],
        [  0, 355,  75, 381]]
    )
    labels=np.array([1, 1, 1, 1, 1])
    segs=[
        [[80, 256, 306, 249, 302, 374, 77, 375]],
        [[398,242,416,249,430,249,447,236,456,248,432,263,413,264,394,254]],
        [[527, 196, 616, 195, 645, 318, 503, 319]],
        [[75, 459, 303, 454, 302, 485, 72, 486]],
        [[0, 355, 75, 358, 73, 381, 0, 378]]
    ]

    draw_img('init.jpg', im.copy(), boxes, segs)
    im, boxes, segs, labels = auth(im, boxes, segs, labels)
    draw_img('transform.jpg', im.copy(), boxes, segs)
if __name__ == '__main__':
    main()