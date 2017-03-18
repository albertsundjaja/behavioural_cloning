from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D

def NVIDIA_SDC(model):
	#convolutional 5x5 kernel, 24 filter, strides 2
	model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), activation="relu", padding="valid"))
	#convolutional 5x5 kernel, 36 filter, strides 2
	model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), activation="relu", padding="valid"))
	#convolutional 5x5 kernel, 48 filter, strides 2
	model.add(Conv2D(48, kernel_size=(5, 5), strides=(2,2), activation="relu", padding="valid"))
	#convolutional 3x3 kernel, 64 filter, strides 2
	model.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), activation="relu", padding="valid"))
	#convolutional 3x3 kernel, 64 filter, strides 2
	model.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), activation="relu", padding="valid"))
	#flatten
	model.add(Flatten())
	#fullcon 1164
	model.add(Dense(1164))
	#fullcon 100
	model.add(Dense(100))
	#fullcon 50
	model.add(Dense(50))
	#fullcon 10
	model.add(Dense(10))
	#final output
	model.add(Dense(1))

	return model

