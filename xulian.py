# 数据集分类后的目录
import itertools
import os

import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from tensorflow.keras import optimizers, models, layers
from tensorflow.python.keras.models import load_model

base_dir = 'D:\\BaiduNetdiskDownload\\train\\train1\\'
# # 训练、验证、测试数据集的目录
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.summary()


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)
# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')

history = model.fit(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)
model.save('./data/cats_and_dogs_small_100.h5')

# def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap='pink', normalize=False):
#     tp = cm[0][0]
#     tn = cm[1][1]
#     fp = cm[1][0]
#     fn = cm[0][1]
#     acc = ((tp+tn)/(tp+tn+fp+fn))
#     recall = ((tp) / (tp + fn))
#     precision = ((tp) / (tp + fp))
#     F1score = (2 * precision * recall / (precision + recall))
#     print("精确值:" + str(acc, 1))
#     print("回召值:" + str(recall, 1))
#     print("准确值:" + str(precision, 1))
#     print("F1score:" + str(F1score))
#
#     if cmap is None:
#         plt.get_cmap('Greens')
#     plt.figure(figsize=(10, 10))
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#
#     if target_names is not None:
#         tick_marks = np.arange(len(target_names))
#         plt.xticks(tick_marks, target_names)
#         plt.yticks(tick_marks, target_names)
#
#     if normalize:
#         cm = cm.astype('float32') / cm.sum(axis=1)
#         cm = np.round(cm, 2)
#
#     thresh = cm.max() / 1.5 if normalize else cm.max() / 2
#     for i, j in itertools.product(range(cm.shape[0]),
#                                   range(cm.shape[1])):
#         if normalize:
#             plt.text(j, i, "{:0.2f}".format(cm[i, j]),
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")
#         else:
#             plt.text(j, i, "{:,}".format(cm[i, j]),
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
#     plt.show()
#
# test_datagen = ImageDataGenerator(rescale=1. / 255)
#
# test_generator = test_datagen.flow_from_directory(
#         test_dir,
#         target_size=(150, 150),
#         batch_size=20,
#         shuffle=False,
#         class_mode='categorical'
#     )
# y_predit = load_model('./data/cats_and_dogs_small_2.h5').predict(test_generator, batch_size=50, verbose=1)
# y_predit = np.argmax(y_predit,axis=1)
# y_true=test_generator.classes
# cm = confusion_matrix(y_true=y_true,y_pred=y_predit)
# target = ['cats', 'dogs']
# plot_confusion_matrix(cm, normalize=True, target_names=target)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



