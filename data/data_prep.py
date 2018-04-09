import os, sys
import math
import pandas as pd
from shutil import copyfile
from sklearn.utils import shuffle

TRAIN_VALIDATION_SPLIT = 0.8

path = os.getcwd() + '/'

labels = pd.read_csv(path + 'labels.csv')
labels = shuffle(labels)
breeds = labels['breed']

for breed in breeds:
    if not os.path.isdir(path + 'train/' + breed):
        os.makedirs(path + 'train/' + breed)
    if not os.path.isdir(path + 'valid/' + breed):
        os.makedirs(path + 'valid/' + breed)

for breed in breeds:
	img_ids = labels[labels['breed'] == breed] ['id'].values
	num_data_pt = len(img_ids)
	num_train = math.ceil(num_data_pt * TRAIN_VALIDATION_SPLIT)
	
	for i in range(num_data_pt):
		if i < num_train:
			folder = 'train/'
		else: 
			folder = 'valid/'
		img = img_ids[i]

		src  = path + 'training_images/' + img + '.jpg'
		dest = path + folder + breed + '/' + img + '.jpg'

		copyfile(src, dest)

sys.exit()