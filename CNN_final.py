# Import necessary libraries
import tensortlow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
#x_train = x_train / 255.0
#x_test = x_test / 255.0

# Define a function to add noise to an image based on a random number
def add_noise(image):
    row, col = image.shape
    R = np.random.uniform()
    if R < 0.25:
        # Gaussian noise
        noise = np.random.normal(loc=0.0, scale=0.1, size=(row, col))
        noisy_image = np.clip(image + noise, 0.0, 1.0)
    elif R < 0.5:
        # Salt and Pepper noise
        amount = np.random.uniform(0.01, 0.1)
        noisy_image = add_salt_pepper_noise(image, amount)
    else:
        # No noise
        noisy_image = image

    return noisy_image

# Define a function to add salt and pepper noise to an image
def add_salt_pepper_noise(image, amount):
    row, col = image.shape
    s_vs_p = 0.5
    out = np.copy(image)

    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = 0

    return out

# Apply the noise function to the training and test data
x_train_noisy = np.array([add_noise(image) for image in X_train])
#x_test_noisy = np.array([add_noise(image) for image in x_test])

# Create a TensorFlow dataset from the noisy training set
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_noisy, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size=512)


# Preprocess the data
x_train_noisy = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
x_train_noisy = x_train_noisy.astype('float32')
X_test = X_test.astype('float32')
x_train_noisy /= 255
X_test /= 255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Create the neural network model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
#model.fit(X_train, y_train, batch_size=128, epochs=1, verbose=1, validation_data=(X_test, y_test))

#Train the model and plot the values of loss and accuracy for each epoch
history = model.fit(x_train_noisy, y_train, batch_size=512, epochs=10, verbose=1, validation_data=(X_test, y_test))


# Evaluate the model on the test set
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Making predictions using our trained model
predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1)


# Display some predictions on test data
fig, axes = plt.subplots(ncols=10, sharex=False,
			 sharey=True, figsize=(20, 4))
for i in range(10):
	axes[i].set_title(predictions[i])
	axes[i].imshow(X_test[i], cmap='gray')
	axes[i].get_xaxis().set_visible(False)
	axes[i].get_yaxis().set_visible(False)
plt.show()


# Plot the values of loss and accuracy for each epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()

    out[coords] = 0

    return out

# Apply the noise function to the training and test data
x_train_noisy = np.array([add_noise(image) for image in x_train])
#x_test_noisy = np.array([add_noise(image) for image in x_test])

# Create a TensorFlow dataset from the noisy training set
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_noisy, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size=512)


# Preprocess the data
x_train_noisy = x_train_noisy.reshape(x_train_noisy.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
x_train_noisy = x_train_noisy.astype('float32')
X_test = X_test.astype('float32')
x_train_noisy /= 255
X_test /= 255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Create the neural network model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
#model.fit(X_train, y_train, batch_size=128, epochs=1, verbose=1, validation_data=(X_test, y_test))

#Train the model and plot the values of loss and accuracy for each epoch
history = model.fit(x_train_noisy, y_train, batch_size=512, epochs=10, verbose=1, validation_data=(X_test, y_test))

# Evaluate the model on the test set
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Making predictions using our trained model
predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1)


# Display some predictions on test data
fig, axes = plt.subplots(ncols=10, sharex=False,
			 sharey=True, figsize=(20, 4))
for i in range(10):
	axes[i].set_title(predictions[i])
	axes[i].imshow(X_test[i], cmap='gray')
	axes[i].get_xaxis().set_visible(False)
	axes[i].get_yaxis().set_visible(False)
plt.show()


# Plot the values of loss and accuracy for each epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()

model.summary()

report = classification_report(np.argmax(y_test, axis=1), predictions)
print(report)
