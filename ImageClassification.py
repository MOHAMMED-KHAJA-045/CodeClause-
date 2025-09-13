# Image Classifier for Automobiles vs. Other Classes
import tensorflow as tf # type: ignore
from tensorflow.keras import datasets, layers, models # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np


# 1. Loading and Preparing Dataset from (CIFAR-10)

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalize pixel values to [0,1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Class labels
cifar10_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

# Create new labels for 'car' vs. 'not a car'
# 'automobile' is at index 1 in the class_names list.
car_index = 1
y_train_binary = np.where(y_train == car_index, 1, 0)
y_test_binary = np.where(y_test == car_index, 1, 0)

# Reshape the binary labels to match the expected input shape
y_train_binary = y_train_binary.reshape(-1)
y_test_binary = y_test_binary.reshape(-1)


# 2. Build CNN Model

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    # The final layer has 1 neuron for binary classification
    layers.Dense(1, activation='sigmoid')
])


# 3. Compile the Model.

model.compile(optimizer='adam',
              # Use binary_crossentropy for binary classification
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])


# 4. Training the Model.

history = model.fit(x_train, y_train_binary, epochs=10,
                    validation_data=(x_test, y_test_binary))

# 5. Evaluating the Model.

test_loss, test_acc = model.evaluate(x_test, y_test_binary, verbose=2)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")


# 6. Plotting Accuracy & Loss :

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()