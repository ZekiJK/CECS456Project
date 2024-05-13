import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32


train_dir = 'D:/CECS456Project/data/natural_images' #pathway


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))


for layer in base_model.layers:
    layer.trainable = False

#classification model on top of the base VGG16 model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(8, activation='softmax')  # Change 8 to the number of classes in your dataset
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#generate trainning and validation data batches from directories
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'  # Use subset for training data
)

validation_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation' 
)


history = model.fit(
    train_gen,
    epochs=10,
    validation_data=validation_gen
)


model.save('vgg16_model.h5') #saved trained model


predictions = model.predict(validation_gen) #make predictions on validation data


first_image_prediction = predictions[0]
predicted_class_index = np.argmax(first_image_prediction)
predicted_class = train_gen.class_indices
print("Predicted class index:", predicted_class_index)
print("Predicted class:", list(predicted_class.keys())[list(predicted_class.values()).index(predicted_class_index)])


validation_loss, validation_accuracy = model.evaluate(validation_gen)
print("Validation Loss:", validation_loss)
print("Validation Accuracy:", validation_accuracy)
