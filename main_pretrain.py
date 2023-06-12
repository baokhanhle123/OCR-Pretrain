"""Setup for PaddleOCR"""
import cv2
from paddleocr import PaddleOCR,draw_ocr
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory

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

img_path = 'test_img.jpg'
ori_img=cv2.imread(img_path,0)
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

# result = ocr.ocr(img_path, cls=True)
result = ocr.ocr(img_path, cls=True, det=True, rec=False)
boxes = result[0]

"""Predict + Visualize"""

# draw result
for i, box in enumerate(boxes):  
  # Convert the box coordinates to integers
  box = [(int(x), int(y)) for x, y in box]

  # Draw the bounding box on the image
  cv2.rectangle(img, box[0], box[2], (255, 0, 0), 1)
  # Save the bounding box coordinates as image coordinates
  x1, y1 = box[0]
  x2, y2 = box[2]
  # p_img=img[y1:y2,x1:x2].copy()
  p_img=img[box[0][1]:box[2][1], box[0][0]:box[2][0]].copy()

  p_img = Image.fromarray(p_img)
  print(detector.predict(p_img))

  
  # Predict the text
  #p_img = Image.fromarray(p_img)
 
  # Save the image
  #p_img.save(f'p_img{i}.jpg')





# save img as result.jpg
# cv2.imwrite('result.jpg', img)