import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

#  constants
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32

# Path
train_dir = 'D:/CECS456Project/data/natural_images'

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


train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'  # Use subset for training data
)

# Load validation data from directory
validation_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'  # Use subset for validation data
)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(8, activation='softmax')  # Change 8 to the number of classes in your dataset
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=validation_gen
)

# Save the model
model.save('my_model.h5')
loaded_model = tf.keras.models.load_model('my_model.h5')

# Evaluate the loaded model on the validation set
validation_loss, validation_accuracy = loaded_model.evaluate(validation_gen)
print("Validation Loss:", validation_loss)
print("Validation Accuracy:", validation_accuracy)

# Make predictions on new data
predictions = loaded_model.predict(validation_gen)

# Example: Print the predicted class for the first image in the validation set
first_image_prediction = predictions[0]
predicted_class_index = np.argmax(first_image_prediction)
predicted_class = train_gen.class_indices
print("Predicted class index:", predicted_class_index)
print("Predicted class:", list(predicted_class.keys())[list(predicted_class.values()).index(predicted_class_index)])
