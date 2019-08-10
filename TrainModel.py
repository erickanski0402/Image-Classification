import os
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Now to build your own working network
#   maybe keep the generated data though

train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                target_size = (64,64),
                                                batch_size = 32,
                                                class_mode = 'binary')
testing_set = test_datagen.flow_from_directory('data/test',
                                                target_size = (64,64),
                                                batch_size = 32,
                                                class_mode = 'binary')

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 10, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

model.fit_generator(training_set,
                    steps_per_epoch = 20000,
                    epochs = 5,
                    validation_data = testing_set,
                    validation_steps = 5000)

print('Would you like to export this model (y/n): ')
response = input()

if response is 'y':
    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('model.h5')
    print('Model saved to disk!')
print('Finished')
