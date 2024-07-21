import pytesseract
import numpy as np
import cv2  # OpenCV

image_path = './invoice.jpg'
img = cv2.imread(image_path)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# !apt-get install tesseract-ocr-sqi Albanian

text = pytesseract.image_to_string(rgb, lang='sqi')
print(text)

# !wget -O ./tessdata/sqi.traineddata https://github.com/tesseract-ocr/tessdata/blob/main/sqi.traineddata?raw=true
# Code below reads text using Trained Albanian Data
# config_tesseract = '--tessdata-dir tessdata'
# text = pytesseract.image_to_string(rgb, lang='sqi', config=config_tesseract)
# print(text)