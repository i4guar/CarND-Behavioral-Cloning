import csv
import cv2
import numpy as np
from PIL import Image
import sklearn
import math




#paths
csv_path = '/opt/data/driving_log.csv'
img_folder_path = '/opt/data/IMG/'


samples = []

# read csv log data
with open(csv_path,'r') as f:
    reader = csv.reader(f)
    for line in reader:
        # append a column in row indicating if image is flipped (1: is not flipped and -1 that it should be flipped)
        unflipped = line + [1]
        flipped = line + [-1]
        samples.append(unflipped)
        samples.append(flipped)        
      

    
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    
from sklearn.utils import shuffle
# generator yielding data samples in batch size
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # generatore never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset + batch_size]
            images = []
            angles = []


            for batch_sample in batch_samples:
                # steering measurement
                steering_center = float(batch_sample[3])

                # adjust steering angle for side camera images
                correction = 0.2 # tuning parameter
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read images
                img_center = np.asarray(Image.open(batch_sample[0])) # if PIL not working use cv2.imread
                img_left = np.asarray(Image.open(batch_sample[1]))
                img_right = np.asarray(Image.open(batch_sample[2]))

                # deciding if sample should be flipped
                if batch_sample[-1] < 0:
                    # add flipped image to dataset
                    # flipping image and argument data
                    images.extend([np.fliplr(img_center), np.fliplr(img_left), np.fliplr(img_right)])
                    angles.extend([-steering_center, -steering_left, -steering_right])
                else:
                    # add unflipped data set
                    images.extend([img_center, img_left, img_right])
                    angles.extend([steering_center, steering_left, steering_right])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
  
    
    
    



# architecture http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, AveragePooling2D, Dropout
model = Sequential()
#crop image
model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(160,320,3)))
# Normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5))

# downsampling making model faster equivalent to resize the image to half the width and height
#model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=24,kernel_size=(5,5),strides=(2,2), activation='relu'))
model.add(Conv2D(filters=36,kernel_size=(5,5),strides=(2,2), activation='relu'))
model.add(Conv2D(filters=48,kernel_size=(5,5),strides=(2,2), activation='relu'))
model.add(Conv2D(filters=64,kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=64,kernel_size=(3,3), activation='relu'))


model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))



# mean square loss because I am want a steering value and no classification
model.compile(loss='mse', optimizer='adam')

# visualizing loss
from keras.models import Model
import matplotlib.pyplot as plt

history_object = model.fit_generator(train_generator, 
            steps_per_epoch=np.ceil(len(train_samples)/batch_size),
            validation_data=validation_generator,
            validation_steps=np.ceil(len(validation_samples)/batch_size),
            epochs=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


model.save('model.h5')