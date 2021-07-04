import tensorflow as tf
import glob
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
tf.__version__

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
training_set = train_datagen.flow_from_directory('brain_tumor_dataset/training_set',  target_size = (64, 64),   batch_size = 32, class_mode = 'binary')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('brain_tumor_dataset/test_set', target_size = (64, 64),batch_size = 32, class_mode = 'binary')

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) #save or ready
#print(training_set.shape)
cnn.fit(x = training_set, epochs = 15) #training(1 iteration consists of  1 batch no of iteration=3000/32 )
print('testing')
cnn.evaluate(x = test_set)
print('Prediction')
test_image = image.load_img('brain_tumor_dataset/Y44.jpg', target_size = (64, 64))
#test_image = image.load_img('brain_tumor_dataset/N7.jpg', target_size = (64, 64))
plt.imshow(test_image)
plt.title('Test Brain Image'), plt.xticks([]), plt.yticks([])
plt.show()
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
#training_set.class_indices
if result[0][0] == 1:
    prediction = 'brain tumor is present'
else:
    prediction = 'brain tumor is not present'
print(prediction)
#print(result)


