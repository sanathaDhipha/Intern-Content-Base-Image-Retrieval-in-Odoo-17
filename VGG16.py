import numpy as np
import h5py
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

class VGGnet:
    def __init__(self):
        self.input_shape=(224, 224, 3)
        self.weight="imagenet"
        self.pooling="max"
        self.model=VGG16(
            input_shape=(self.input_shape),
            weights=self.weight,
            pooling=self.pooling,
            include_top=False     
        )
        self.model.predict(np.zeros((1, 224, 224, 3)))

    def extract_features(self, path):
        img=image.load_img(path, target_size=(self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        img=image.img_to_array(img)
        img=np.expand_dims(img, axis=0)
        img=preprocess_input(img)
        feature=self.model.predict(img)
        feature=feature[0]/np.linalg.norm(feature[0])
        return feature