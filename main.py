import numpy as np
from SegmentPage import segment_into_lines

"""Setup for VietOCR"""
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_transformer')
# config['weights'] = './weights/transformerocr.pth'
config['cnn']['pretrained']=False
config['device'] = 'cpu'

detector = Predictor(config)

"""Segmentation"""
#Open image and segment into lines
line_img_array, coordinates = segment_into_lines('./test_img.jpg')

print(type(line_img_array[0]))
print(line_img_array[0].shape)

# Draw line_image_array[0] on a white background and save it as a new image
# import cv2
# cv2.imwrite('line_image.jpg',line_img_array[1])

img = Image.open('./test_img.jpg')

for i, line in enumerate(line_img_array):
    try:
        """
        img = f'line_image_{i}.jpg'
        cv2.imwrite(img,line)
        img = Image.open(img)
        """

        img = Image.fromarray(line)
        print(detector.predict(img))
        print(coordinates[i])
    except:
        pass


# img = Image.open(img)
"""
img = 'line_image_1.jpg'
img = Image.open(img)
print(detector.predict(img))
"""
