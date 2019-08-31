import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.optimizers import RMSprop


# Loading data
dataset = pd.read_csv("PATH/train.csv")

label = dataset['label'].values
y = label.reshape(-1, 1)

# Encoding labels
enc = preprocessing.OneHotEncoder()
enc.fit(y)
onehotlabels = enc.transform(y).toarray()

# Creating training data
all = np.array(dataset)
X = []
for i in all:
    X.append(i[1:])

X = np.array(X)

# Preprocessing Data
X = tf.keras.utils.normalize(X, axis=1)

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, onehotlabels, test_size=0.2, random_state=0)

# Building model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(250, activation='relu', input_dim=784),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0003), metrics=['acc'])

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=2)

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
plt.plot(epochs, acc, color="blue", label='Acc')
plt.plot(epochs, val_acc, color="green", label='Val_Acc')
plt.title("Training and validation accuracy")
plt.legend()

plt.figure()

plt.plot(epochs, loss, color="blue", label='Loss')
plt.plot(epochs, val_loss, color="green", label='Val_Loss')
plt.title("Training and validation loss")
plt.legend()
plt.show()
