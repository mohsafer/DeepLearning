

# Import necessary libraries
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

# Define a function to add noise to the images
def add_noise(images, noise_type='gaussian', mean=0, stddev=0.1):
    if noise_type == 'gaussian':
        # Add Gaussian noise to the images
        noise = tf.random.normal(shape=tf.shape(images), mean=mean, stddev=stddev, dtype=tf.float32)
        noisy_images = images + noise
    elif noise_type == 'salt_and_pepper':
        # Add salt-and-pepper noise to the images
        noise = tf.random.uniform(shape=tf.shape(images), minval=0, maxval=1, dtype=tf.float32)
        noisy_images = tf.where(noise < 0.05, 0.0, images)
        noisy_images = tf.where(noise > 0.95, 1.0, noisy_images)
    else:
        raise ValueError('Invalid noise type')
        return noisy_images


# Apply noise to the image based on the value o
R = random.random()

if R < 0.25:
   x_train_noisy = add_noise(X_train, noise_type='gaussian', mean=0, stddev=0.1)
   print(f"[Random Value]:{R:.10f} [gaussian]")
elif R < 0.5:
   x_train_noisy = add_noise(X_train, noise_type='salt_and_pepper', mean=0, stddev=0.1)
   print(f"[Random Value]:{R:.10f} [Salt and Pepper]")
else:
    x_train_noisy = X_train
    print(f"[Random Value]:{R:.10f} [Normal]")


# Add noise to the training set
#x_train_noisy = add_noise(x_train, noise_type='gaussian', mean=0, stddev=0.1)

# Create a TensorFlow dataset from the noisy training set
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_noisy, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size=32)




# Preprocess the data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Create the neural network model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=128, epochs=1, verbose=1, validation_data=(X_test, y_test))

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
