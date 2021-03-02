from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


classifier=Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))# 32 kernels of 3 * 3 size, Input_shae is of type colourful images . so RGB
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dropout(0.15))
classifier.add(Dense(units=5,activation='softmax')) # Multi Class Classification
#classifier.add(Dense(units=1,activation='sigmoid')) # For binary Classification
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy' , metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen =ImageDataGenerator(rescale=1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
training_set=train_datagen.flow_from_directory(r'D:\Sasi_pr\Projects\CNN_Classifier\Dataset_Classifier',target_size=(64,64),batch_size=32)
testing_set=test_datagen.flow_from_directory(r'D:\Sasi_pr\Projects\CNN_Classifier\Dataset_Classifier',target_size=(64,64),batch_size=32)
model = classifier.fit_generator(training_set,steps_per_epoch=800,epochs=2,validation_data=testing_set,validation_steps=100)

model.history
"""
{
'val_loss': [0.00929802842438221, 0.002544076880440116],
 'val_accuracy': [1.0, 0.9990285038948059],
 'loss': [0.16788231441788845, 0.012336105469959929],
 'accuracy': [0.94037586, 0.9971637]
}
"""

classifier.save("multiClassClassifierCNN_Model.h5")
print("Saved to disk")

import numpy as np
from keras.preprocessing import image
from keras.models import load_model
test_image = image.load_img('D:/Sasi_pr/Projects/images.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
model=load_model('multiClassClassifierCNN_Model.h5')
result = model.predict(test_image)
training_set.class_indices
print(result)
if result[0][0] == 0:
    prediction = 'BlackPlum Tree'
    print(prediction)
elif result[0][0] == 1:
    prediction = 'Coconut Tree'
    print(prediction)
elif result[0][0] == 2:
    prediction = 'MS Dhoni'
    print(prediction)
elif result[0][0] == 3:
    prediction = 'Rohit Sharma'
    print(prediction)
elif result[0][0] == 4:
    prediction = 'Sasi Kumar'
    print(prediction)