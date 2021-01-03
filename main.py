""" Who's that Pokemon?
	Here we write a machine learning program to classify pictures of Pokemon.
	We use machine learning python libraries including TensorFlow and Keras.
	Results will typically be displayed in Plotly and Matplotlib.

	Written by Shaun Miller using the Kaggle dataset https://www.kaggle.com/thedagger/pokemon-generation-one.
	This program follows some of the tutorial from the TensorFlow website: https://www.tensorflow.org/tutorials/images/classification.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import pathlib
import sys

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #silence some of the tf warnings

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


#Download and explore the dataset

def get_local_data(file_path, val_split=0.2, batch_size = 32, height = 180, width = 180):
	#returns training and validation datasets from a local dataset

	data_dir = pathlib.Path(file_path)
	print('Number of JPEG in dataset:',len(list(data_dir.glob('*/*.jpg')))) #print the total amount of images
	train_ds = tf.keras.preprocessing.image_dataset_from_directory(
		data_dir,
		validation_split = val_split, #100-val_split% used for training, val_split% for validation
		subset="training",
		seed=710,
		image_size=(height, width),
		batch_size=batch_size)

	val_ds = tf.keras.preprocessing.image_dataset_from_directory(
		data_dir,
		validation_split = val_split,
		subset="validation",
		seed=710,
		image_size=(height, width),
		batch_size=batch_size)

	class_names = train_ds.class_names
	print('Pokemon in set:', class_names[:3], ' ... ', class_names[-3:])

	return train_ds, val_ds

def configure_ds(train_ds, val_ds):
	AUTOTUNE = tf.data.experimental.AUTOTUNE
	train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
	val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
	return train_ds, val_ds

def create_model(num_classes = 5, height = 180, width = 180):

	#add data augmentation for more accurate results
	data_augmentation = keras.Sequential(
		[layers.experimental.preprocessing.RandomFlip("horizontal",
		    input_shape=(height,width,3)),
			layers.experimental.preprocessing.RandomRotation(0.1),
			layers.experimental.preprocessing.RandomZoom(0.1)]
		)

	model = Sequential([

		data_augmentation,
		#normalize model with rescaling
		layers.experimental.preprocessing.Rescaling(1./255, input_shape=(height, width, 3)),

		layers.Conv2D(16, 3, padding='same', activation='relu'),
		layers.MaxPooling2D(),
		layers.Conv2D(32, 3, padding='same', activation='relu'),
		layers.MaxPooling2D(),
		#layers.Conv2D(64, 3, padding='same', activation='relu'), #uncomment for entire data set
		#layers.MaxPooling2D(), #uncomment for entire data set
		layers.Dropout(0.15),
		layers.Flatten(),
		#layers.Dense(128, activation='relu'), #uncomment for entire data set
		layers.Dense(num_classes)
	])

	#compile the model
	model.compile(optimizer='adam',
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy'])

	return model

def train_model(model, train_ds, val_ds, epochs= 12):
	history = model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=epochs
		)
	return history, epochs

def make_prediction(model, picture_path, height = 180, width = 180):

	img = keras.preprocessing.image.load_img(
    	picture_path, target_size=(height, width)
	)
	img_array = keras.preprocessing.image.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0) # Create a batch

	predictions = model.predict(img_array)
	score = tf.nn.softmax(predictions[0])

	result_string = "Likely {} with {:.2f}% confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
	#written by Shaun Miller
	img_to_print = Image.open(picture_path)
	draw = ImageDraw.Draw(img_to_print)
	font = ImageFont.truetype("arial.ttf", 100)
	draw.text((0, 0),result_string,(255,255,255),font=font)
	img_to_print.show()


if __name__ == "__main__":

	#Retreive dataset
	train_ds, val_ds = get_local_data('dataset_popular')
	class_names = train_ds.class_names

	#Train the Model
	train_ds, val_ds = configure_ds(train_ds, val_ds)
	model = create_model()
	print(model.summary())
	history, epochs = train_model(model, train_ds, val_ds)

	#Test the Model
	for picture_path in sys.argv[1:]: make_prediction(model, picture_path = picture_path)
