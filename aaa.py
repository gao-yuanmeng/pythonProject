import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image# abc

base_dir = 'D:\\BaiduNetdiskDownload\\train\\train1'
train_dir = os.path.join(base_dir, 'train/')
validation_dir = os.path.join(base_dir, 'validation/')

train_cats_dir = os.path.join(train_dir, 'cats')  # 训练猫图片集
train_dogs_dir = os.path.join(train_dir, 'dogs')  # 训练狗图片集
validation_cats_dir = os.path.join(validation_dir, 'cats')  # 测试猫图片集
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # 测试狗图片集


batch_size = 128  # 样本数
epochs = 15  # 学习步长
IMG_HEIGHT = 150  # 图片高度
IMG_WIDTH = 150  # 图片宽度

num_cats_tr = len(os.listdir(train_cats_dir))  # 猫图片测试集的数量:1000
num_dogs_tr = len(os.listdir(train_dogs_dir))  # 狗图片测试集的数量:1000

num_cats_val = len(os.listdir(validation_cats_dir))  # 猫图片测试集的数量:500
num_dogs_val = len(os.listdir(validation_dogs_dir))  # 猫图片测试集的数量:500

total_train = num_cats_tr + num_dogs_tr  # 训练数据数量:2000
total_val = num_cats_val + num_dogs_val  # 测试数据数量:1000

# 生成训练数据集和验证数据集
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
validation_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

# 在为训练和验证图像定义生成器之后，flow_from_directory方法从磁盘加载图像，应用重新缩放，并将图像调整到所需的尺寸。
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')


# 创建模型：该模型由三个卷积块组成，每个卷积块中有一个最大池层。有一个完全连接的层，上面有512个单元，由relu激活功能激活。

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1)
])

# 编译模型：这边选择ADAM优化器和二进制交叉熵损失函数。传递metrics参数查看每个训练时期的训练和验证准确性。交叉熵损失函数
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# 训练模型：使用ImageDataGenerator类的fit_generator方法来训练网络。
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

# 可视化训练结果
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#保存模型
model.save_weights('./save_weights/my_save_weights')

#本程序存在过拟合现象

filePath = base_dir + '/tmp/content/'
keys = os.listdir(filePath)

for fn in keys:
    # 对图片进行预测
    # 读取图片
    path = base_dir + '/tmp/content/' + fn
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
    # 在第0维添加维度变为1x150x150x3，和我们模型的输入数据一样
    x = np.expand_dims(x, axis=0)
    # np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组，我们一次只有一个数据所以不这样也可以
    images = np.vstack([x])
    # batch_size批量大小，程序会分批次地预测测试数据，这样比每次预测一个样本会快。因为我们也只有一个测试所以不用也可以
    classes = model.predict(images, batch_size=10)
    print(classes[0])

    if classes[0] > 0:
        print(fn + " is a dog")

    else:
        print(fn + " is a cat")

