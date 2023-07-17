import keras.callbacks
import tensorflow as tf
from keras.layers import Dense, BatchNormalization
from keras.activations import relu, sigmoid
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
import h5py
import numpy as np
import matplotlib.pyplot as plt

#this might be
#etest
def load_dataset():
    train_dataset = h5py.File(r"D:\programs\python\soft_math1\deep_learning\datasets/train_catvnoncat.h5")
    train_set_x_org = np.array(train_dataset['train_set_x'][:])
    train_set_y_org = np.array(train_dataset['train_set_y'][:])

    test_dataset = h5py.File(r"D:\programs\python\soft_math1\deep_learning\datasets/test_catvnoncat.h5")
    test_set_x_org = np.array(test_dataset['test_set_x'][:])
    test_set_y_org = np.array(test_dataset['test_set_y'][:])

    classes = np.array(test_dataset['list_classes'][:])

    #train_set_y_org = np.reshape(train_set_y_org, (1, train_set_y_org.shape[0]))
    #test_set_y_org = np.reshape(test_set_y_org, (1, test_set_y_org.shape[0]))

    return train_set_x_org, train_set_y_org, test_set_x_org, test_set_y_org, classes

train_set_x_org, train_set_y_org, test_set_x_org, test_set_y_org, classes = load_dataset()

index = 25
plt.imshow(train_set_x_org[index])
plt.show()
print("label",train_set_y_org[index])


train_set_x_flatten = train_set_x_org.reshape((train_set_x_org.shape[0], -1))
test_set_x_flatten = test_set_x_org.reshape((test_set_x_org.shape[0], -1))

print("flat shape",train_set_x_flatten.shape)

train_set_x_flatten = train_set_x_flatten/255.0
test_set_x_flatten = test_set_x_flatten/255.0

model = keras.models.load_model("training_model/saved_model-05.h5")

pred = model.predict(test_set_x_flatten)

model = tf.keras.models.Sequential(
    layers=[
tf.keras.layers.Dense(64, activation='relu', input_shape=(12288, )),
tf.keras.layers.Dense(64, activation='relu'),
Dense(1, activation='sigmoid')
]
)

#model.load_weights()
check_point =keras.callbacks.ModelCheckpoint(
    filepath="training_model/saved_model-{epoch:02d}.h5",
    monitor='val_loss',
    save_best_only=True,
)
early_stoping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              min_delta=.02,
                                              patience=5,
                                              )

model.compile(loss=BinaryCrossentropy(),optimizer=Adam(learning_rate=.001),
              metrics=['accuracy'])
history = model.fit(x=train_set_x_flatten , y=train_set_y_org ,
          validation_data=(test_set_x_flatten, test_set_y_org),
                    batch_size=32, epochs=50, callbacks=[check_point, early_stoping])

model.save("training_model/test_binary.h5")
model.summary()
