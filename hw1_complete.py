#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import layers, Sequential

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image


print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


##

def build_model1():
  """Fully-connected: Flatten + 3 Dense(128, leaky_relu) + Dense(10, no activation)."""
  model = Sequential([
      layers.Flatten(input_shape=(32, 32, 3)),
      layers.Dense(128, activation='leaky_relu'),
      layers.Dense(128, activation='leaky_relu'),
      layers.Dense(128, activation='leaky_relu'),
      layers.Dense(10),
  ])
  return model

def build_model2():
  """Conv2D stack matching test expected layers and param counts."""
  model = Sequential([
      layers.Conv2D(32, (3, 3), strides=2, padding='same', activation='relu', input_shape=(32, 32, 3)),
      layers.BatchNormalization(),
      layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.Flatten(),
      layers.Dense(10),
  ])
  return model

def build_model3():
  """SeparableConv2D stack (Sequential so layer list matches test; no InputLayer)."""
  model = Sequential([
      layers.SeparableConv2D(32, (3, 3), strides=2, padding='same', activation='relu', input_shape=(32, 32, 3)),
      layers.BatchNormalization(),
      layers.SeparableConv2D(64, (3, 3), strides=2, padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.Flatten(),
      layers.Dense(10),
  ])
  return model

def build_model50k():
  """Small CNN with <= 50k parameters for CIFAR-10."""
  model = Sequential([
      layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
      layers.BatchNormalization(),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.Flatten(),
      layers.Dense(10),
  ])
  return model

# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':

  ########################################
  ## Load the CIFAR10 data set and split into train / validation / test
  (train_images, train_labels), (test_images, test_labels) = \
      tf.keras.datasets.cifar10.load_data()
  train_labels = train_labels.squeeze()
  test_labels = test_labels.squeeze()
  train_images = train_images / 255.0
  test_images = test_images / 255.0

  # Split training set into training and validation (e.g. last 10% for validation)
  n_train = len(train_images)
  n_val = n_train // 10
  val_images = train_images[-n_val:]
  val_labels = train_labels[-n_val:]
  train_images = train_images[:-n_val]
  train_labels = train_labels[:-n_val]
  # Now: (train_images, train_labels), (test_images, test_labels), (val_images, val_labels)

  CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

  ########################################
  ## Build and train model 1 (30 epochs)
  model1 = build_model1()
  model1.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
  model1.fit(train_images, train_labels, epochs=30, batch_size=64, verbose=1,
             validation_data=(val_images, val_labels))
  _, train_acc1 = model1.evaluate(train_images, train_labels, verbose=0)
  _, val_acc1 = model1.evaluate(val_images, val_labels, verbose=0)
  _, test_acc1 = model1.evaluate(test_images, test_labels, verbose=0)
  print('Model1 — train acc: {:.4f}, val acc: {:.4f}, test acc: {:.4f}'.format(train_acc1, val_acc1, test_acc1))

  ## Build, compile, and train model 2 (30 epochs)
  model2 = build_model2()
  model2.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
  model2.fit(train_images, train_labels, epochs=30, batch_size=64, verbose=1,
             validation_data=(val_images, val_labels))
  _, train_acc2 = model2.evaluate(train_images, train_labels, verbose=0)
  _, val_acc2 = model2.evaluate(val_images, val_labels, verbose=0)
  _, test_acc2 = model2.evaluate(test_images, test_labels, verbose=0)
  print('Model2 — train acc: {:.4f}, val acc: {:.4f}, test acc: {:.4f}'.format(train_acc2, val_acc2, test_acc2))

  ## Build, compile, and train model 3 (30 epochs)
  model3 = build_model3()
  model3.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
  model3.fit(train_images, train_labels, epochs=30, batch_size=64, verbose=1,
             validation_data=(val_images, val_labels))
  _, train_acc3 = model3.evaluate(train_images, train_labels, verbose=0)
  _, val_acc3 = model3.evaluate(val_images, val_labels, verbose=0)
  _, test_acc3 = model3.evaluate(test_images, test_labels, verbose=0)
  print('Model3 — train acc: {:.4f}, val acc: {:.4f}, test acc: {:.4f}'.format(train_acc3, val_acc3, test_acc3))

  ########################################
  ## Classify a custom image (test_image_classname.png or .jpg)
  import glob
  test_image_paths = glob.glob('test.png') + glob.glob('test.jpg')
  if test_image_paths:
    path = test_image_paths[0]
    test_img = np.array(keras.utils.load_img(path, grayscale=False, color_mode='rgb', target_size=(32, 32)))
    test_img = (test_img / 255.0).astype(np.float32)
    test_img_batch = np.expand_dims(test_img, axis=0)
    pred_logits2 = model2.predict(test_img_batch, verbose=0)
    pred_class2 = int(np.argmax(pred_logits2[0]))
    pred_logits3 = model3.predict(test_img_batch, verbose=0)
    pred_class3 = int(np.argmax(pred_logits3[0]))
    print('Custom image "{}" — Model2 predicted: {}, Model3 predicted: {}'.format(
        path, CIFAR10_CLASSES[pred_class2], CIFAR10_CLASSES[pred_class3]))
  else:
    print('No test_image_*.png or test_image_*.jpg found; skip custom image classification.')

  ########################################
  ## Build, compile, and train model50k; save to best_model.h5
  model50k = build_model50k()
  model50k.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
  model50k.fit(train_images, train_labels, epochs=15, batch_size=64, verbose=1,
               validation_data=(val_images, val_labels))
  model50k.save('best_model.h5')
  _, test_acc50k = model50k.evaluate(test_images, test_labels, verbose=0)
  print('Model50k — test accuracy: {:.4f}'.format(test_acc50k))

  # plt.show()  # Commented out so script runs without intervention
