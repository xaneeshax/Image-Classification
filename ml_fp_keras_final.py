## **1. Data Loading**
"""

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator')
else:
  print(gpu_info)

! pip install kaggle --upgrade

! mkdir ~/.kaggle/

! cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets download -d puneet6060/intel-image-classification

! mkdir dataset

! unzip intel-image-classification.zip -d dataset

! ls dataset/

! ls dataset/seg_pred/seg_pred/


"""## **2. EDA**"""

# Imports
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import statistics
import plotly.graph_objects as go

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

import cv2 as cv
from google.colab.patches import cv2_imshow
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

# how an image looks
I = np.asarray(PIL.Image.open("dataset/seg_pred/seg_pred/10004.jpg"))
im = PIL.Image.fromarray(np.uint8(I))
im

# define parameters for Keras to load data into variables
image_size = (150, 150)
img_height = 150
img_width = 150
batch_size = 32

# create the datasets using Keras DataSet object
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/seg_train/seg_train/",
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/seg_test/seg_test/",
    image_size=image_size,
    batch_size=batch_size,
)

im_classes = {
  0 : 'Building',
  1 : 'Forest',
  2 : 'Glacier',
  3 : 'Mountain',
  4 : 'Sea',
  5 : 'Street',
}
class_labels = list(im_classes.values())
num_classes = len(class_labels)

# EDA - randomly sampling 9 images across all the categories
plt.figure(figsize=(14, 8))
for images, labels in train_ds.take(1):
    for i in range(18):
        ax = plt.subplot(3, 6, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(im_classes[int(labels[i])])
        plt.axis("off")

# EDA - see image altering techniques (can be used for image segmentation purposes later)
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

# EDA - look at proportion of each class inside the training dataset
images_per_class = {}
for folder in os.listdir("dataset/seg_train/seg_train"):
  images_per_class[folder.capitalize()] = len(os.listdir(f'dataset/seg_train/seg_train/{folder}'))

print(images_per_class)

dataset_size = sum(images_per_class.values())

for key, index in images_per_class.items():
  images_per_class[key] = round(images_per_class[key] / dataset_size,3)

print(images_per_class)

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = images_per_class.keys()
sizes = images_per_class.values()
explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05) 

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        colors=['#e0a760','#95a399','#1bde50','#668cad','#32c5db','#3777bd'], 
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

# EDA - look at proportion of each class inside the validation dataset
images_per_class_validation = {}
for folder in os.listdir("dataset/seg_test/seg_test"):
  images_per_class_validation[folder.capitalize()] = len(os.listdir("dataset/seg_test/seg_test/" + str(folder)))

print(images_per_class_validation)

dataset_size_val = sum(images_per_class_validation.values())

for key, index in images_per_class_validation.items():
  images_per_class_validation[key] = round(images_per_class_validation[key] / dataset_size_val,3)

print(images_per_class_validation)

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = images_per_class_validation.keys()
sizes = images_per_class_validation.values()
explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05) 

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        colors=['#e0a760','#95a399','#1bde50','#668cad','#32c5db','#3777bd'], 
        shadow=True, startangle=90)
ax1.axis('equal')

plt.show()

# EDA - for each class (folder), look at the average distribution of RGB values across all pixels in the images
colors = ('b', 'g', 'r')

for folder in os.listdir('dataset/seg_train/seg_train'):
  blue_array = [0 for i in range(0, 256)]
  green_array = [0 for i in range(0, 256)]
  red_array = [0 for i in range(0, 256)]
  
  count = 0
  for file in os.listdir('dataset/seg_train/seg_train/' + folder):
    count += 1
    img = cv.imread(f'dataset/seg_train/seg_train/{folder}/{file}')
    for i, col in enumerate(colors):
      hist = cv.calcHist([img], [i], None, [256], [0,256])
      if (i == 0):
        blue_array += hist
      elif (i ==1):
        green_array += hist
      else: #i==2:
        red_array += hist
  
  blue_array = blue_array / count
  green_array = green_array / count
  red_array = red_array / count
  
  fig = go.Figure()
  print(np.array(blue_array).shape)

  fig.add_trace(go.Scatter(x=[num for num in range(0,256)], y=[num[0] for num in blue_array],
                    mode='lines+markers', line = dict(width=4, color='blue'),
                    name='Blue Channel'))
  fig.add_trace(go.Scatter(x=[num for num in range(0,256)], y=[num[0] for num in red_array],
                    mode='lines+markers', line = dict(width=4, color='red'),
                    name='Red Channel'))
  fig.add_trace(go.Scatter(x=[num for num in range(0,256)], y=[num[0] for num in green_array],
                    mode='lines+markers', line = dict(width=4, color='green'),
                    name='Green Channel'))
  fig.update_layout(title_text=folder.upper(), template='plotly_white')
  fig.update_layout(xaxis_title= 'Pixel Range', yaxis_title= 'Counts')
  fig.update_layout(autosize=False, width=700, height=400)
  fig.update_layout(yaxis_range=[0,1000])
  fig.update_layout(
    title={
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
  fig.show()

# EDA - for each class, look at the average contours count, where a counter represents a boundary of an object
objects = {}
for folder in os.listdir("dataset/seg_train/seg_train"):
  countours_total = []
  count = 0
  for file in os.listdir("dataset/seg_train/seg_train/" + folder):
    count += 1
    img = cv.imread(f"dataset/seg_train/seg_train/{folder}/{file}")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Binarizes the image
    ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
    # Identifying contours -> The boundries of objects
    # Commonly used for object recognition
    # RETR_LIST -> returns all contours, some return only external ones or hierarchial ones
    # APPROX -> how we want to approximate the countours
    # CHAIN_APPROX -> Compresses a line to endpoints
    countours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    countours_total.append(len(countours))

  countours_average = sum(countours_total) / count
  countours_sd = statistics.stdev(countours_total)
  objects[folder.capitalize()] = [countours_average, countours_sd]

  # interpretation: the higher the number of average countours, 
  # the more "entropy" (the picture has less uniformity)

objects # Represents mean and standard deviation

classes= list(objects.keys())
colors = ['#3777bd','#95a399','#1bde50','#668cad','#e0a760','#32c5db']
averages = [mean for mean, std in objects.values()]

fig = go.Figure([go.Bar(x=classes, y=averages, marker_color=colors)])
fig.update_traces(marker_line_color='#aeb3b8', marker_line_width=1.5, opacity=0.8)
fig.update_layout(title_text='Average Countours by Class')
fig.update_layout(xaxis_title= 'Class Name', yaxis_title= 'Number of Contours')
fig.update_layout(autosize=False, width=700, height=400)
fig.update_layout(template='plotly_white')
fig.update_layout(yaxis_range=[0,600])
fig.update_layout(
    title={
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()

"""## **3. Defining Metric Functions**"""

def get_preds_and_trues(model_used, ds):
  trues = []
  preds = []
  for input, labels in ds:
    batch_predictions = model_used.predict(x=input)
    for pred in batch_predictions:
      preds.append(np.argmax(tf.nn.softmax(pred)))
    for label in labels:
      trues.append(int(label))
  return trues, preds

def get_probas(model_used, ds):
  trues = []
  preds = []
  probas = []
  for input, labels in ds:
    batch_predictions = model_used.predict(x=input)
    for pred in batch_predictions:
      preds.append(np.argmax(tf.nn.softmax(pred)))
      probas.append([float(proba) for proba in pred])
    for label in labels:
      trues.append(int(label))
  return trues, preds, probas

def get_confusion_df(model, val_ds):
  # column is what it actually is, row is prediction
  # labels running on top is predicted
  # labels running on side is actual
  # 73 images actually class 1 predicted class 6

  trues, preds = get_preds_and_trues(model, val_ds)
  confusion = confusion_matrix(y_true = trues, y_pred = preds)
  confusion_df = pd.DataFrame(confusion, index = class_labels, columns = class_labels)
  
  return confusion_df

def get_class_report(trues, preds):
  class_report = classification_report(y_true = trues, y_pred = preds, target_names=class_labels)
  print(class_report)

# use confusion matrix df to calculate TPR, FPR per class
# https://medium.datadriveninvestor.com/confusion-matric-tpr-fpr-fnr-tnr-precision-recall-f1-score-73efa162a25f

def get_fpr(df, class_name):
  tp = df[class_name][class_name]
  fp = df.loc[class_name].sum() - tp
  fn = df[class_name].sum() - tp
  tn = sum(df.sum()) - fp - fn + tp

  return fp / (tn+fp)

def get_tpr(df, class_name):
  tp = df[class_name][class_name]
  fp = df.loc[class_name].sum() - tp
  fn = df[class_name].sum() - tp

  return tp / (fp + fn)

# Precision, Recall and F1 Scores
def get_precision(df, class_name):
  tp = df[class_name][class_name]
  fp = df.loc[class_name].sum() - tp

  return tp / (tp + fp)

def get_recall(df, class_name):
  tp = df[class_name][class_name]
  fn = df[class_name].sum() - tp

  return tp / (tp + fn)

def get_f1(df, class_name):
  precision = get_precision(class_name)
  recall = get_recall(class_name)

  return 2 * (precision * recall / (precision + recall))

def plot_heamtmap(confusion_matrix, name): 
  # figure
  fig, ax = plt.subplots(figsize=(11, 9))

  # plot heatmap
  sns.heatmap(confusion_matrix, annot = True, cmap="Blues", vmin= 0, vmax=550, 
              square=True, linewidth=0.8, fmt='g', 
              xticklabels=class_labels, yticklabels=class_labels,
              annot_kws={'fontsize':15})
  # xticks
  ax.xaxis.tick_top()

  # axis labels
  plt.xlabel('PREDICTED')
  plt.ylabel('ACTUAL')

  # title
  title = f'{name} Performance\n'.upper()
  plt.title(title, loc='left')
  plt.show()

def plot_roc_curve(trues, preds, probs, name):

  unique_classes = list(set(preds))
  preds = label_binarize(preds, classes=unique_classes)
  probs = np.array(probs)

  # Compute ROC curve and ROC area for each class
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(len(unique_classes)):
      fpr[i], tpr[i], _ = roc_curve(preds[:, i], probs[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

  # Compute micro-average ROC curve and ROC area
  fpr["micro"], tpr["micro"], _ = roc_curve(preds.ravel(), probs.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

  # First aggregate all false positive rates
  all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(unique_classes))]))

  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i]) 

  # Finally average it and compute AUC
  mean_tpr /= num_classes

  fpr["macro"] = all_fpr
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

  # Plot all ROC curves
  info = {
    'micro' : 'Micro-Average ROC curve',
    'macro' : 'Macro-Average ROC curve',
    0: 'Building ROC Curve',
    1: 'Forest ROC Curve',
    2: 'Glacier ROC Curve',
    3: 'Mountain ROC Curve',
    4: 'Sea ROC Curve',
    5: 'Street ROC Curve'
  }

  # Create traces
  fig = go.Figure()
  line_type = lambda label : dict(width=4, dash='dash') if label in ('micro', 'macro') else dict(width=2)
  
  for label in info.keys():
    fig.add_trace(go.Scatter(x=fpr[label], y=tpr[label],mode='lines', 
                            line = line_type(label),
                            name=f'{info[label]} (AUC = {round(roc_auc[label],2)})'))

  fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Random Classifier',line=dict(width=4, dash='dash', color='black')))
  fig.update_layout(title_text=f'{name} ROC Curve', template='plotly_white')
  fig.update_layout(xaxis_title= 'False Positive Rate', yaxis_title= 'True Positive Rate')
  fig.update_layout(autosize=False, width=1000, height=600)
  fig.show()

def plot_conv_layers(img_path, model):

  successive_outputs = [layer.output for layer in model.layers[1:]]
  visualization_model = tf.keras.models.Model(inputs = model.input, 
                                              outputs = successive_outputs)
  #Load the input image
  img = load_img(img_path, target_size=(150, 150))
  x = img_to_array(img)                           
  x = x.reshape((1,) + x.shape)
  x /= 255.0

  # Obtain all intermediate representations for the image.
  successive_feature_maps = visualization_model.predict(x)

  # Retrieve are the names of the layers, so can have them as part of our plot
  layer_names = [layer.name for layer in model.layers]
  for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if 'conv' in layer_name:
      
      # Plot feature maps for the conv layers
      n_features = feature_map.shape[-1]  # number of features 

      fig = plt.figure(figsize = (18,10))

      i_vals = []
      for i in range(n_features):
        x  = feature_map[0, :, :, i]
        if x.sum() > 0:
          i_vals.append(i)
      
      for i in range(len(i_vals)):
        x = feature_map[0, :, :, i_vals[i]]
        x -= x.mean()
        x /= x.std ()
        x *=  64
        x += 128
        x  = np.clip(x, 0, 255).astype('uint8')
        
        fig.add_subplot(1, len(i_vals), i+1)
        plt.axis('off')
        plt.imshow(np.array(x), cmap='terrain')

def plot_val_data(model_history):

  acc = model_history.history['accuracy']
  val_acc = model_history.history['val_accuracy']

  loss = model_history.history['loss']
  val_loss = model_history.history['val_loss']

  epochs_range = range(epochs)

  plt.figure(figsize=(16, 6))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()

"""## **4. Modeling - Feed Forward Neural Networks**"""

num_classes = 6

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Flatten(),
  layers.Dense(10),
  layers.Activation('relu'),
  layers.Dense(1),
  layers.Activation('softmax'),
])

rms = keras.optimizers.RMSprop()
model.compile(optimizer=rms,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

epochs=10

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  batch_size=batch_size,
)

ffnn_df = get_confusion_df(model, val_ds)
plot_heamtmap(ffnn_df, 'Feed Forward NN')

ff_trues, ff_preds, ff_probas = get_probas(model, val_ds)
plot_roc_curve(ff_trues, ff_preds, ff_probas, 'Feed Forward Neural Network')

get_class_report(ff_trues, ff_preds)

"""## **5.Modeling - Convolutional Neural Networks**"""

model2 = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model2.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model2.summary()

epochs=10

history2 = model2.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  batch_size=batch_size,
)

cnn_df = get_confusion_df(model2, val_ds)
plot_heamtmap(cnn_df, 'Convolutional NN')

cnn_trues, cnn_preds, cnn_probas = get_probas(model2, val_ds)
plot_roc_curve(cnn_trues, cnn_preds, cnn_probas, 'Convolutional NN')

get_class_report(cnn_trues, cnn_preds)

plot_val_data(history2)

img_path = "dataset/seg_train/seg_train/forest/10007.jpg" 
plot_conv_layers(img_path, model2)

img_path = "dataset/seg_train/seg_train/glacier/10064.jpg" 
plot_conv_layers(img_path, model2)

img_path = "dataset/seg_train/seg_train/mountain/18186.jpg" 
plot_conv_layers(img_path, model2)

"""## **6.Modeling - CNN with Dropout Layers**"""

def create_dropout_model():
  model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.25),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.25),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.25),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(num_classes)
  ])
  return model

dropout_model = create_dropout_model()

dropout_model.summary()

batch_size = 32
epochs = 10

dropout_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history_dropout = dropout_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  batch_size=batch_size,
)

do_cnn_df = get_confusion_df(dropout_model, val_ds)
plot_heamtmap(do_cnn_df, 'Convolutional NN with Dropout')

do_cnn_trues, do_cnn_preds, do_cnn_probas = get_probas(dropout_model, val_ds)
plot_roc_curve(do_cnn_trues, do_cnn_preds, do_cnn_probas, 'Convolutional NN with Dropout')

img_path = "dataset/seg_train/seg_train/forest/10007.jpg" 
plot_conv_layers(img_path, dropout_model)

img_path = "dataset/seg_train/seg_train/mountain/18186.jpg" 
plot_conv_layers(img_path, dropout_model)

folder = 'glacier'
file = '100.jpg'
img_path = f"dataset/seg_train/seg_train/{folder}/{file}"
plot_conv_layers(img_path, dropout_model)

get_class_report(do_cnn_trues, do_cnn_preds)

plot_val_data(history_dropout)

def create_extra_dropout_model():
  model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.4),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.4),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.4),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(num_classes)
  ])
  return model

extra_dropout_model = create_extra_dropout_model()

batch_size = 32
epochs = 10

extra_dropout_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history_extra_dropout = extra_dropout_model.fit(
  train_ds_splitColors,
  validation_data=val_ds,
  epochs=epochs,
  batch_size=batch_size,
)

do_cnn_sc_df = get_confusion_df(dropout_model, val_ds)
plot_heamtmap(do_cnn_sc_df, 'Convolutional NN with Dropout and Split Colors Filter')

do_cnn_sc_trues, do_cnn_sc_preds, do_cnn_sc_probas = get_probas(dropout_model, val_ds)
plot_roc_curve(do_cnn_sc_trues, do_cnn_sc_preds, do_cnn_sc_probas, 'Convolutional NN with Dropout and Split Colors Filter')

"""## **7.Modeling - Convolution with Reducing Nodes in Hidden Layers**"""

model3 = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(16, activation='relu'),
  layers.Dense(6)
])

model3.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model3.summary()

epochs=10

history3 = model3.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  batch_size=batch_size,
)

down_cnn_trues, down_cnn_preds, down_cnn_probas = get_probas(model3, val_ds)
plot_roc_curve(down_cnn_trues, down_cnn_preds, down_cnn_probas, 'Convolutional NN Downwards')

get_class_report(down_cnn_trues, down_cnn_preds)

plot_val_data(history3)

"""## **8. Modeling - Transfer Learning with ImageNet**"""

base_model = keras.applications.Xception(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False,
)  # Do not include the ImageNet classifier at the top.

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=(150, 150, 3))

# Pre-trained Xception weights requires that input be normalized
# from (0, 255) to a range (-1., +1.), the normalization layer
# does the following, outputs = (inputs - mean) / sqrt(var)
norm_layer = keras.layers.experimental.preprocessing.Normalization()
mean = np.array([127.5] * 3)
var = mean ** 2
# Scale inputs to [-1, +1]
x = norm_layer(inputs)
norm_layer.set_weights([mean, var])

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(6)(x)
transfer_learning_model = keras.Model(inputs, outputs)

transfer_learning_model.summary()

transfer_learning_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 10
transfer_learning_model.fit(train_ds, epochs=epochs, validation_data=val_ds)

imagenet_df = get_confusion_df(transfer_learning_model, val_ds)
plot_heamtmap(imagenet_df, 'Transfer Learning with ImageNet')

imnet_trues, imnet_preds, imnet_probas = get_probas(transfer_learning_model, val_ds)
plot_roc_curve(imnet_trues, imnet_preds, imnet_probas, 'Transfer Learning with ImageNet')

im_classes

metrics = pd.DataFrame(imnet_probas, columns=im_classes.values())

metrics['trues'] = [im_classes[val] for val in imnet_trues]

metrics['preds'] = [im_classes[pred] for pred in imnet_preds]

glacier_df = metrics[metrics['trues'] == 'Glacier']
mt_as_glacier = glacier_df[glacier_df['preds'] == 'Mountain']

mt_as_glacier['difference'] = mt_as_glacier['Mountain'] - mt_as_glacier['Glacier']

mt_as_glacier.head()

mt_as_glacier.describe()

get_class_report(imnet_trues, imnet_preds)

# Image Net with Bilateral Filter Images
transfer_learning_model.fit(train_ds_bilateral, epochs=epochs, validation_data=val_ds)

imagenet_bilat_df = get_confusion_df(transfer_learning_model, val_ds)
plot_heamtmap(imagenet_bilat_df, 'Transfer Learning with ImageNet and Bilateral Filter')

imagenet_bilat_trues, imagenet_bilat_preds, imagenet_bilat_probas = get_probas(transfer_learning_model, val_ds)
plot_roc_curve(imagenet_bilat_trues, imagenet_bilat_preds, imagenet_bilat_probas, 'Transfer Learning with ImageNet and Bilateral Filter')

get_class_report(imagenet_bilat_trues, imagenet_bilat_preds)

"""## **9. Modeling - Image Augmentation**"""

def load_data(augment_type_name):
    
    IMAGE_SIZE = (150,150)
    dataset = 'dataset/seg_train/seg_train'
    new_dataset = f'dataset/seg_train_{augment_type_name}/seg_train_{augment_type_name}'
    os.makedirs(new_dataset)

    # Iterate through each folder corresponding to a category
    for folder in os.listdir(dataset):
        print(folder)
        os.makedirs((os.path.join(new_dataset, folder)))
        
        # Iterate through each image in our folder
        for file in tqdm(os.listdir(os.path.join(dataset, folder))):
            
            # Get the path name of the image
            img_path = os.path.join(os.path.join(dataset, folder), file)
            
            # Open and resize the img
            image = cv.imread(img_path)
            
            # image = cv.bilateralFilter(image, 10, 30, 30)
            # image = cv.Canny(image, 125, 175)
            b,g,r = cv.split(image)
            image = cv.merge([g,b,r])
            image = cv.resize(image, IMAGE_SIZE) 

            new_path = os.path.join(os.path.join(new_dataset, folder), file)
            cv.imwrite(new_path, image)

load_data('augmentedWithBilateralFilter')

load_data('canny')

load_data('splitColors')

# how an image looks
I = np.asarray(PIL.Image.open("dataset/seg_train_splitColors/seg_train_splitColors/street/1000.jpg"))
im = PIL.Image.fromarray(np.uint8(I))
im

# how an image looks
I = np.asarray(PIL.Image.open("dataset/seg_train/seg_train/street/1000.jpg"))
im = PIL.Image.fromarray(np.uint8(I))
im

train_ds_bilateral = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/seg_train_augmentedWithBilateralFilter/seg_train_augmentedWithBilateralFilter/",
    image_size=image_size,
    batch_size=batch_size,
)

train_ds_canny = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/seg_train_canny/seg_train_canny/",
    image_size=image_size,
    batch_size=batch_size,
)

train_ds_splitColors = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/seg_train_splitColors/seg_train_splitColors/",
    image_size=image_size,
    batch_size=batch_size,
)

history_dropout_bilateral = dropout_model.fit(
  train_ds_bilateral,
  validation_data=val_ds,
  epochs=epochs,
  batch_size=batch_size,
)

history_dropout_canny = dropout_model.fit(
  train_ds_canny,
  validation_data=val_ds,
  epochs=epochs,
  batch_size=batch_size,
)

history_dropout_splitColors = dropout_model.fit(
  train_ds_splitColors,
  validation_data=val_ds,
  epochs=epochs,
  batch_size=batch_size,
)

do_bilateral_cnn_df = get_confusion_df(dropout_model, val_ds)
plot_heamtmap(do_bilateral_cnn_df, 'Convolutional NN with Dropout and Image Augmentation Using Bilateral Filter Operation')

do_canny_cnn_df = get_confusion_df(dropout_model, val_ds)
plot_heamtmap(do_canny_cnn_df, 'Convolutional NN with Dropout and Image Augmentation Using Canny Operation')

do_splitColors_cnn_df = get_confusion_df(dropout_model, val_ds)
plot_heamtmap(do_splitColors_cnn_df, 'Convolutional NN with Dropout and Image Augmentation Using Split Colors Operation')

do_bilateral_cnn_trues, do_bilateral_cnn_preds, do_bilateral_cnn_probas = get_probas(dropout_model, val_ds)
plot_roc_curve(do_bilateral_cnn_trues, do_bilateral_cnn_preds, do_bilateral_cnn_probas, 'Convolutional NN with Dropout and Image Augmentation Using Bilteral Filter Operation')

do_canny_cnn_trues, do_canny_cnn_preds, do_canny_cnn_probas = get_probas(dropout_model, val_ds)
plot_roc_curve(do_canny_cnn_trues, do_canny_cnn_preds, do_canny_cnn_probas, 'Convolutional NN with Dropout and Image Augmentation Using Canny Operation')

do_splitColors_cnn_trues, do_splitColors_cnn_preds, do_splitColors_cnn_probas = get_probas(dropout_model, val_ds)
plot_roc_curve(do_splitColors_cnn_trues, do_splitColors_cnn_preds, do_splitColors_cnn_probas, 'Convolutional NN with Dropout and Image Augmentation Using Canny Operation')

get_class_report(do_bilateral_cnn_trues, do_bilateral_cnn_preds)

get_class_report(do_canny_cnn_trues, do_canny_cnn_preds)

get_class_report(do_splitColors_cnn_trues, do_splitColors_cnn_preds)

(train_images, train_labels), (test_images, test_labels) = load_data()



# for image augmentation use:
https://learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/
https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/

# for custom data augmentation use:
https://www.tensorflow.org/tutorials/images/data_augmentation#custom_data_augmentation
# EXACT EXAMPLE
https://www.analyticsvidhya.com/blog/2020/11/extending-the-imagedatagenerator-keras-tensorflow/ 

# for transfer leaning we want:
Use Convolutional Nets as Feature Extractor
# example from tensorflow that looks easier
https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub#download_the_headless_model
# keras guide example with code
https://keras.io/guides/transfer_learning/#build-a-model

# for metrics
# roc:
https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/\

# resnet
