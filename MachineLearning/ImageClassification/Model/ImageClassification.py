
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
classifier = Sequential()

# update the dataset path

Training_dataset  = 'P:/ssc/ImageClassification/data/training_set-20180226T160955Z-001/training_set'
Test_dataset = 'P:/ssc/ImageClassification/data/test_set-20180226T160745Z-001/test_set'

# adding convolution

classifier.add(Conv2D(32,(3,3), input_shape= (64,64,3), activation='relu'))

# pooling

classifier.add(MaxPooling2D(pool_size=(2,2)))

# adding second convolution and pool
classifier.add(Conv2D(32,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# flattening

classifier.add(Flatten())

# full connection

classifier.add(Dense(units= 128, activation= 'relu'))

classifier.add(Dense(units= 1, activation= 'sigmoid'))

classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

tain_datagen = ImageDataGenerator(rescale=1./255, shear_range= 0.2, zoom_range= 0.2, horizontal_flip= True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = tain_datagen.flow_from_directory(Training_dataset,
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
test_set = test_datagen.flow_from_directory(Test_dataset,
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

classifier.fit_generator(training_set,
steps_per_epoch = 180,
epochs = 25,
validation_data = test_set,
validation_steps = 120)


