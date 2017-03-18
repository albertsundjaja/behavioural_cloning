import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Cropping2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from nvidia_sdc import NVIDIA_SDC

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            #correction parameter to use the multiple cameras
            correction = 0.15
            images = []
            angles = []

            for batch_sample in batch_samples:
                name_center = 'data_collection/IMG/' + batch_sample[0].split('\\')[-1]
                name_left = 'data_collection/IMG/' + batch_sample[1].split('\\')[-1]
                name_right = 'data_collection/IMG/' + batch_sample[2].split('\\')[-1]
                steering_center = float(batch_sample[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                center_image = cv2.imread(name_center)
                left_image = cv2.imread(name_left)
                right_image = cv2.imread(name_right)
                
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                angles.append(steering_center)
                angles.append(steering_left)
                angles.append(steering_right)
                #augment data with flipped version
                images.append(cv2.flip(center_image,1))
                images.append(cv2.flip(left_image,1))
                images.append(cv2.flip(right_image,1))
                angles.append(steering_center * -1)
                angles.append(steering_left * -1)
                angles.append(steering_right * -1)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)

lines = []
with open('data_collection/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)
train_generator = generator(train_samples, batch_size=8)
validation_generator = generator(validation_samples, batch_size=8)

model = Sequential()
#crop our image
model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=(160,320,3)))
#normalisation and mean centering
model.add(Lambda(lambda x: x / 255.0 - 0.5))
#load our NVIDIA model
model = NVIDIA_SDC(model)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch= 
            len(train_samples), validation_data=validation_generator, 
            validation_steps=len(validation_samples), epochs=1)

model.save('model.h5')