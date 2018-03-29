import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from keras.preprocessing import image


class DataGetter():
    def __init__(self, img_width, img_height, img_channels):
        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = img_channels

    def load_images(self, images_path, labels_path=""):
        image_names = next(os.walk(images_path))[2]
        images = np.zeros((len(image_names), self.img_width, self.img_height,
                           self.img_channels), dtype=np.float32)
        corresp_dict = {}
        for i, name in tqdm(enumerate(image_names), total=len(image_names)):
            corresp_dict[i] = name
            image_path = images_path + name
            img = image.load_img(image_path,
                                 target_size=(self.img_width, self.img_height))
            images[i] = image.img_to_array(img)

        if labels_path:
            labels = np.zeros(len(image_names))
            label_file = pd.read_csv(labels_path, sep='\t', header=None)
            label_file.columns = ['name', 'label', 'a', 'b', 'c', 'd']
            labels = label_file['label'].values
            return images, labels, corresp_dict

        return images, corresp_dict
