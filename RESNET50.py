import numpy as np
import h5py
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

class RESNET50:
    def __init__(self):
        self.input_shape=(224, 224, 3)
        self.weight="imagenet"
        self.resnet_model=ResNet50(
            input_shape=(self.input_shape),
            weights=self.weight,
            include_top=True
        )
        self.model=Model(self.resnet_model.input, self.resnet_model.get_layer("avg_pool").output)
    
    def extract_features(self, path):
        img=image.load_img(path, target_size=(self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        img=image.img_to_array(img)
        img=np.expand_dims(img, axis=0)
        img=preprocess_input(img)
        feature=self.model.predict(img)
        feature=feature[0]/np.linalg.norm(feature[0])
        return feature