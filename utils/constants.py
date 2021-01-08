

import os

root_dir = ’C:\root_directory\’
data_dir = root_dir + ’\data\’
fruit_models_dir = root_dir + ’\ fruit_models \’
labels_file = root_dir + ’\ utils \ labels ’

training_images_dir = ’\Fruit -Images -Dataset \Training 
test_images_dir = ’\Fruit -Images -Dataset \ Test ’


with open(labels_file) as f:
  labels = f.readlines()
  num_classes = len(labels) + 1
number_train_images = \trainingImageCount
number_test_images = \testImageCount
