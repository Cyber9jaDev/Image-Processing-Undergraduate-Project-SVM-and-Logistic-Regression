import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from skimage.feature import graycomatrix, graycoprops, graycomatrix
from skimage import io, color, img_as_ubyte
from skimage.feature import local_binary_pattern
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class params:
  data_path = '../Dataset'
  bad_images = '../Dataset/Bad'
  good_images = '../Dataset/Good'
  csv_files = '../Dataset/csv_files'

dataset = pd.DataFrame(
  {
    'images': [f"{params.bad_images}/{x}" for x in os.listdir(params.bad_images)] + [f"{params.good_images}/{x}" for x in os.listdir(params.good_images)],
    'label': [0]*len(os.listdir(params.bad_images)) + [1]*len(os.listdir(params.good_images))
  }
)
dataset.head()
dataset['label'].value_counts(normalize = True)

X = dataset['images']
y = dataset['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
y_train.value_counts(normalize = True)
y_test.value_counts(normalize = True)

def ExtractFeatures(image_ids):
  # Features to be extracted
  features = {
    "image_id": [],
    "average_hue":[],
    "average_saturation": [],
    "average_value" : [],
    "area": [],
    "x":[], "y":[],
    "w":[], "h":[],
    "aspect_ratio": [],
    "correlation": [],
    "energy":[],
    "contrast":[],
    "homogeneity":[]
  }

  for image_id in tqdm(image_ids, desc="Processing", unit="images"):
    features["image_id"].append(image_id)
    image = cv2.imread(image_id)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Color Features
    average_color = np.mean(image, axis=(0, 1))
    color_histogram = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    dominant_color = np.argmax(color_histogram)
    color_variance = np.var(image)

    # 1. Average color in HSV channels (Hue, Saturation, Value)
    h, s, v,_ = cv2.mean(hsv)
    features["average_hue"].append(h)
    features["average_saturation"].append(s)
    features["average_value"].append(v)

    # 3. Area of the pepper (assuming some segmentation is done beforehand)
    # Replace this with your segmentation method (e.g., Otsu's thresholding)
    # This is a placeholder for demonstration purposes
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
      largest_area = cv2.contourArea(contours[0])
      for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > largest_area:
          largest_area = area
    else:
      largest_area = 0
      
    features["area"].append(largest_area)
    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    (x, y, w, h) = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h

    features["w"].append(w)
    features["h"].append(h)
    features["x"].append(x)
    features["y"].append(y)
    features["aspect_ratio"].append(aspect_ratio)

    glcm = graycomatrix(gray_image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')

    features["correlation"].append(correlation.mean())
    features["energy"].append(energy.mean())
    features["contrast"].append(contrast.mean())
    features["homogeneity"].append(homogeneity.mean())

  return pd.DataFrame(features)

## Extracting Training Dataset
features_df = ExtractFeatures(image_ids = X_train.values)

