import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions


def make_prediction(model, image_path, image_size):
    img = image.load_img(image_path, target_size=image_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    predictions = model.predict(img)
    decoded_predictions = decode_predictions(predictions)
    return predictions, decoded_predictions
