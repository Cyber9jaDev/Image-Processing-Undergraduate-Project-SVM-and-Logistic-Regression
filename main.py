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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
from tqdm import tqdm
from sklearn.svm import SVC
import sklearn
import seaborn as sns
import joblib
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

features_df["quality"] = y_train.values
features_df.head()

## Saving Training Dataset
features_df.to_csv(f"{params.csv_files}/Train.csv", index = False)

# ## Extracting Testing Dataset
features_test = ExtractFeatures(image_ids = X_test.values)

features_test["quality"] = y_test.values
features_test.head()

# ## Saving Testing Dataset
features_test.to_csv(f"{params.csv_files}/Test.csv", index = False)


# Read CSV Files
train_df = pd.read_csv(f"{params.csv_files}/Train.csv")
test_df = pd.read_csv(f"{params.csv_files}/Test.csv")

train_df.shape, test_df.shape

train_df.head()

# Exploratory Analysis

test_df['quality'].value_counts()

# Sample data
labels = ['Good Images', 'Bad Images']
counts_train = [282, 82]  # Replace with your actual counts
counts_test = [70, 21]  # Replace with your actual counts

# Calculate percentages
total_count_train = sum(counts_train)
percentages_train = [(count / total_count_train) * 100 for count in counts_train]

total_count_test = sum(counts_test)
percentages_test = [(count / total_count_test) * 100 for count in counts_test]

# Define pleasing colors
colors = ['#66b2ff', '#ff9999']

# Plotting the 2D pie chart with shadow
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Define the amount of explosion (0.1 means 10% of the radius)
explode = (0.1, 0)

ax[0].pie(
  counts_train, 
  labels=labels, 
  autopct=lambda p: '{:.0f} ({:.1f}%)'.format(p * total_count_train / 100, p), 
  startangle=90, 
  colors=colors, 
  wedgeprops={'edgecolor': 'black'}, 
  shadow=True, 
  explode=explode
)
# Add a title
ax[0].set_title('Distribution of Training Images')

ax[1].pie(
  counts_test, 
  labels=labels, 
  autopct=lambda p: '{:.0f} ({:.1f}%)'.format(p * total_count_test / 100, p), 
  startangle=90, 
  colors=colors, 
  wedgeprops={'edgecolor': 'black'}, 
  shadow=True, 
  explode=explode
)
# Add a title
ax[1].set_title('Distribution of Test Images')

# Display the 2D pie chart with shadow
plt.show()

train_df.describe()

plt.figure(figsize=(10,3))
sns.distplot(train_df["average_hue"], color = "green")
plt.title("Average Hue Distribution")
plt.show()

plt.figure(figsize=(10,3))
sns.distplot(train_df["area"], color = "green")
plt.title("Area Distribution")
plt.show()

plt.figure(figsize=(10,3))
sns.distplot(train_df["aspect_ratio"], color = "green")
plt.title("Aspect Ratio Distribution")
plt.show()

sns.distplot(train_df["correlation"], color = "green")
plt.title("Correlation Distribution")
plt.show()

sns.distplot(train_df["energy"], color = "green")
plt.title("Energy Distribution")
plt.show()

sns.distplot(train_df["contrast"], color = "green")
plt.title("Contrast Distribution")
plt.show()

sns.distplot(train_df["homogeneity"], color = "green")
plt.title("Homogeneity Distribution")
plt.show()

# Feature Engineering and Selection
train_df.columns
features = ['average_hue', 'area', 'aspect_ratio', 'correlation', 'energy', 'contrast', 'homogeneity', 'x', 'y', 'w', 'h']
target = ["quality"]

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

X_train.head(3)

plt.figure(figsize = (10,5))

plt.scatter(
  train_df[train_df['quality'].eq(1)]['average_hue'], 
  train_df[train_df['quality'].eq(1)]['homogeneity'],  
  color = 'g', 
  label = 'Good quality'
)

plt.scatter(
  train_df[train_df['quality'].eq(0)]['average_hue'], 
  train_df[train_df['quality'].eq(0)]['homogeneity'],  
  color = 'r', 
  label = 'Bad quality'
)

plt.xlabel('Hue')
plt.ylabel('Homogeneity')
plt.title('Hue vs Homogeneity')
plt.legend()

plt.figure(figsize = (10,5))

plt.scatter(train_df[train_df['quality'].eq(1)]['energy'], train_df[train_df['quality'].eq(1)]['contrast'],  color = 'g', label = 'Good quality')
plt.scatter(train_df[train_df['quality'].eq(0)]['energy'], train_df[train_df['quality'].eq(0)]['contrast'],  color = 'r', label = 'Bad quality')

plt.xlabel('Energy')
plt.ylabel('Contrast')
plt.title('Energy vs Contrast')
plt.legend()

# Feature Scaling (Standardization)
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler instance
scaler = StandardScaler()
scaler.fit(X_train)
# Fit the scaler to the data and transform the data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



model_name = "scaler.joblib"

# Saving it with Joblib's gentle hand,
joblib.dump(scaler, model_name)

X_train


# Logistic Regression
model_log_reg = LogisticRegression()

# Train the model on the training data
model_log_reg.fit(X_train, y_train)

# Make predictions on the test data
pred_log_reg = model_log_reg.predict(X_test)


model_svc = SVC()

# Train the model on the training data
model_svc.fit(X_train, y_train)

# Make predictions on the test data
pred_svc = model_svc.predict(X_test)


# Model Evaluation
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
  import matplotlib.pyplot as plt
  import numpy as np
  import itertools

  accuracy = np.trace(cm) / float(np.sum(cm))
  misclass = 1 - accuracy

  if cmap is None:
    cmap = plt.get_cmap('Blues')

  plt.figure(figsize=(8, 6))
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()

  if target_names is not None:
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  thresh = cm.max() / 1.5 if normalize else cm.max() / 2
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if normalize:
      plt.text(j, i, "{:0.4f}".format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    else:
      plt.text(j, i, "{:,}".format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
  plt.show()
    
# Model Accuracy
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, pred_log_reg)}")
print(f"Support Vector Classifier Accuracy: {accuracy_score(y_test, pred_svc)}")

## Logistic Regression
print(classification_report(y_test, pred_log_reg))

## Logistic Regression
confusion_matrix(y_test, pred_log_reg)

plot_confusion_matrix(
  cm = confusion_matrix(y_test, pred_log_reg), 
  normalize = False, 
  target_names = ['Bad', 'Good'], 
  title = "Confusion Matrix for Logistic Regression Classifier"
)

# SVM Classifier
print(classification_report(y_test, pred_svc))

plot_confusion_matrix(
  cm = confusion_matrix(y_test, pred_svc), 
  normalize = False, 
  target_names = ['Bad', 'Good'], 
  title = "Confusion Matrix for SVM Classifier"
)


# Your model's name and path, oh so dear,
model_name = "model.joblib"

# Saving it with Joblib's gentle hand,
joblib.dump(model_log_reg, model_name)

print(sklearn.__version__)