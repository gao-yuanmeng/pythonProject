import os
import matplotlib.pyplot as plt
from tensorflow.python.keras.api.keras.models import load_model
import numpy as np
from PIL import Image
location = "D:\\BaiduNetdiskDownload\\train\\train1\\test\\dogs\\dog.1507.jpg"
pil = Image.open(location,'r')
plt.imshow(np.asarray(pil))
pil = pil.resize((150,150))
pil = np.array(pil) /255.
pil = pil.reshape(1,150,150,3)
model = load_model('C:\\Users\\GaoYM\\PycharmProjects\\pythonProject\\data\\cats_and_dogs_small_100.h5')
pred = model.predict(pil)

if pred[0][0] >pred[0][1]:
    result = pred[0][0]
    print("检测为猫，概率为",result)
if pred[0][0] <pred[0][1]:
    result = pred[0][1]
    print("检测为狗，概率为",result)
plt.show()