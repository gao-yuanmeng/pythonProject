#导包
import itertools
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.api.keras import optimizers
from tensorflow.python.keras.api.keras import models
from tensorflow.python.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.api import keras
from tensorflow.python.keras.api.keras.layers import Flatten
from tensorflow.python.keras.api.keras.layers import Dense
from tensorflow.python.keras.api.keras.models import load_model
from sklearn.metrics import confusion_matrix

location = 'C:\\Users\\86176\\Desktop\\JavaWork\\pythonProject\\trainningData\\cnndata.h5'

#创建一个cnn模型
def get_model():
    model = models.Sequential()
    conv_base = keras.applications.DenseNet201(weights='imagenet', include_top=False)

    conv_base.trainable = True
    for layers in conv_base.layers[:-5]:
        layers.trainable = False

    model.add(conv_base)

    model.add(keras.layers.GlobalAveragePooling2D())

    model.add(Flatten())
        #全连接层
    model.add(Dense(512,activation="relu"))
    model.add(Dense(2,activation="softmax"))

    model.summary()

    model.compile(loss="binary_crossentropy",
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=["acc"])

    train_ImageDataGenerator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True
    )

    train_Generator = train_ImageDataGenerator.flow_from_directory(
        "D:\\mechine learning data\\trainer\\train",
        target_size=(150,150),
        batch_size=20,
        class_mode = "categorical"
    )

    validation_ImageDataGenerator = ImageDataGenerator(rescale=1. / 255)

    validation_Generator = validation_ImageDataGenerator.flow_from_directory(
        "D:\\mechine learning data\\trainer\\validation",
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical'
    )

    history = mod