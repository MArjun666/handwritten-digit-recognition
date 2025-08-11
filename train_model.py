import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = utils.to_categorical(train_labels)
test_labels = utils.to_categorical(test_labels)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define ModelCheckpoint callback to save the best model
checkpoint = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

# Train the model
print("Starting model training...")
model.fit(train_images, train_labels, epochs=10, batch_size=64,
          validation_data=(test_images, test_labels),
          callbacks=[checkpoint])
print("Model training completed and the best model is saved as model.h5")
