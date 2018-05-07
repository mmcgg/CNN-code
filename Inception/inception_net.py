from keras.applications.inception_v3 import InceptionV3 
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import numpy as np

model = InceptionV3(weights='imagenet')

img_path = 'C:/Users/Spencer/Documents/GitHub/CNN-code/dataset/lfw/train/Paul_Krueger/Paul_Krueger_0001.jpg'

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)

print(features)