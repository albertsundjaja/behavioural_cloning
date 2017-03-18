# Included Files
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* README.md (this file) summarizing the results

# How To
To begin using the model, execute the command
`python drive.py model.h5`
Run the simulator and choose autonomous

# Model Architecture
The model used is the modified version of NVIDIA Self Driving Car Architecture
which can be found [here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

The model consist of 5 convolutional layers and 4 fully connected layers (the original consists of 3 fully connected layers)

The first three convolutional layers use 5x5 kernel with strides of 2, the last two layers use 3x3 kernel with strides of 1. ReLU activation is used to introduce non-linearity

The detail of the model is as follow:
1. 5x5, stride = 2, depth = 24, activation = ReLU, padding = valid
2. 5x5, stride = 2, depth = 36, activation = ReLU, padding = valid 
3. 5x5, stride = 2, depth = 48, activation = ReLU, padding = valid 
4. 3x3, stride = 1, depth = 64, activation = ReLU, padding = valid 
5. 3x3, stride = 1, depth = 64, activation = ReLU, padding = valid 
6. Flatten
7. Fully connected, out 1164
8. Fully connected, out 100
9. Fully connected, out 50
10. Fully connected, out 10
11. Final output, output = steering angle

### Data Preprocessing
The data is normalized and mean-centered using Keras Lambda layer. It is also cropped to remove the scenery part of the image and the hood of the car.

### Data Augmentation
Data is augmented with flipped images, further, the multiple camera approach is used, with steering adjusted by -0.15 and 0.15 for the respective left and right image. Multiple camera approach proves to be effective as it is simulating "recovery driving" process.

### Overfitting 
As the model is trained with 3000 images and 1 epoch, overfitting problem did not arise. Adding dropout to the model proves to be detrimental. The Rubric mentioned that I need to include a dropout, if it is a required criteria to pass the project, then I would need to train using more datas and perhaps making adjustment to the model.

### Model Parameter
The model is optimized using Keras Adam optimizer with setting left at default.

# Training Strategy
I started training using LeNet and VGG model, I also created my own model. But it was unsuccessful. I have tried to add more data to the model. I ended with around 20,000 samples but it doesnt improve the model.

As mentioned above, the final model is taken from NVIDIA Self Driving Car. I hate to say this, but the reasoning on why this model is used as reference is because if it is good enough from them, it should be good enough for me. It turns out that this is the case, I only added another fully connected layer to the model. Further, I only used 3000 images and 1 epoch to train it. NVIDIA's model proves to be very effective.

# Data Collection

Data collection is done by doing two full laps across the track, and making sure the car stays in the model throughout the process. I take 3000 samples from the result, and divided it to train and validation sets. Validation is 20% of the whole samples.

# Further Improvement

The model did not run successfully in track 2. I am currently working to make this possible.




